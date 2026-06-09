"""
Multi-Task Fan-out: 任务分解 + 依赖感知的并行执行

用法:
    from src.multi_task_agent import MultiTaskManager
    mgr = MultiTaskManager(llm, store, embedder_model)
    tasks = mgr.decompose("查A库和B库的歌手数，对比")
    plan = mgr.build_execution_plan(tasks)
    results = mgr.execute_wave(plan[0])  # 并行执行
"""

import json
import os
import operator
from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

SPIDER_DATA = os.path.join(os.path.dirname(__file__), "..", "spider_data", "database")

DECOMPOSE_PROMPT = """分析用户查询，将其分解为独立的子任务，标注依赖关系。

## 输出格式
返回 JSON，包含一个 tasks 数组：
```json
{
  "tasks": [
    {"id": "A", "type": "sql_query", "prompt": "查询concert_singer数据库的歌手数量", "depends": []},
    {"id": "B", "type": "sql_query", "prompt": "查询singer数据库的歌手数量", "depends": []},
    {"id": "C", "type": "synthesize", "prompt": "对比A和B的查询结果，给出结论", "depends": ["A", "B"]}
  ]
}
```

## 任务类型
- **sql_query**: 数据库查询（包含数据库名、查询描述、聚合等）
- **synthesize**: 综合对比前面的结果，生成文字结论（不查询数据库）
- **chart**: 根据前面查询的数据生成图表（bar/pie/line）

## 规则
- 独立的数据库查询放在不同任务中，depends 为空
- 依赖前面查询结果的任务（对比、图表、综合）必须标注 depends
- 任务 prompt 要具体，包含数据库名、查询字段、条件
- 结果会注入到 prompt 中，格式为: [上一个任务的结果是: ...]

只返回 JSON，不要其他内容。"""

SUB_TASK_PROMPT = """你是一个 Text-to-SQL 子任务执行器。只完成分配给你的这一个子任务。

## 任务
{prompt}

## 已有上下文（前面任务的结果）
{context}

## 规则
- 如果任务是 sql_query：调用 search_databases → search_fields → get_table_schema → submit_sql
- 如果任务是 synthesize：综合上下文中的结果，给出文字结论，不需要查询数据库
- 如果任务是 chart：从上下文中提取数据，调用 create_chart 生成图表
- 只做你的任务，不要做多余的查询
- submit_sql 后直接给出结果，不需要额外解释"""


# ---- 数据模型 --------------------------------------------------------------

class TaskResult(TypedDict, total=False):
    task_id: str
    task_type: str
    prompt: str
    result: str
    status: str           # "pending" | "running" | "done" | "failed"


class ExecutionPlan(TypedDict):
    waves: list[list[dict]]     # 每波的任务列表


# ---- MultiTaskManager ------------------------------------------------------

class MultiTaskManager:
    """多任务管理器：分解 → 调度 → 执行 → 聚合。"""

    def __init__(self, llm, store, embedder_model):
        self.llm = llm
        self.store = store
        self.model = embedder_model
        self._results_cache: dict[str, str] = {}

    # ---- 任务分解 ----------------------------------------------------------

    def decompose(self, query: str) -> list[dict]:
        """LLM 分析用户查询，输出任务 DAG。"""
        resp = self.llm.invoke([
            SystemMessage(content=DECOMPOSE_PROMPT),
            HumanMessage(content=query),
        ], temperature=0.1)

        raw = resp.content.strip()
        try:
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            tasks = data.get("tasks", [])
            if tasks:
                return tasks
        except json.JSONDecodeError:
            pass

        # fallback: 单任务
        return [{"id": "A", "type": "sql_query", "prompt": query, "depends": []}]

    # ---- 调度：将 DAG 转为波次 -----------------------------------------------

    def build_execution_plan(self, tasks: list[dict]) -> ExecutionPlan:
        """按依赖关系将任务分组为波次（wave）。
        同一波次内的任务无相互依赖，可并行执行。
        """
        completed: set[str] = set()
        remaining = list(tasks)
        waves = []

        while remaining:
            wave = []
            still_waiting = []
            for t in remaining:
                if all(d in completed for d in t.get("depends", [])):
                    wave.append(t)
                else:
                    still_waiting.append(t)

            if not wave:
                # 死锁：有循环依赖
                break

            waves.append(wave)
            for t in wave:
                completed.add(t["id"])
            remaining = still_waiting

        return {"waves": waves}

    # ---- 执行单个子任务 ----------------------------------------------------

    def execute_single_task(self, task: dict) -> dict:
        """执行一个子任务（可在独立线程/进程中调用）。
        返回: {"task_id": str, "success": bool, "result": str}
        """
        task_id = task["id"]
        task_type = task["type"]
        prompt = task.get("prompt", "")

        # 构建上下文：注入依赖任务的结果
        context_lines = []
        for dep_id in task.get("depends", []):
            if dep_id in self._results_cache:
                # 依赖任务可能已失败，传递其状态
                dep_result = self._results_cache[dep_id]
                if isinstance(dep_result, dict):
                    status = "成功" if dep_result.get("success") else "失败"
                    dep_text = dep_result.get("result", "")
                else:
                    status, dep_text = "成功", str(dep_result)
                context_lines.append(
                    f"任务 {dep_id} (状态: {status}) 的结果:\n{dep_text[:800]}"
                )
        context = "\n\n".join(context_lines) if context_lines else "（无依赖，首次查询）"

        try:
            if task_type == "sql_query":
                result_text = self._run_sql_task(prompt, context)
            elif task_type == "synthesize":
                result_text = self._run_synthesize(prompt, context)
            elif task_type == "chart":
                result_text = self._run_chart_task(prompt, context)
            else:
                result_text = self._run_sql_task(prompt, context)

            self._results_cache[task_id] = {
                "success": True,
                "result": result_text,
            }
            return {"task_id": task_id, "success": True, "result": result_text}

        except Exception as e:
            error_msg = f"任务执行失败: {str(e)[:300]}"
            import traceback
            log = __import__("logging").getLogger("MultiTask")
            log.error("  [%s] %s", task_id, error_msg)
            log.debug("  Traceback: %s", traceback.format_exc()[-500:])

            self._results_cache[task_id] = {
                "success": False,
                "result": error_msg,
            }
            return {"task_id": task_id, "success": False, "result": error_msg}

    def _run_sql_task(self, prompt: str, context: str) -> str:
        """执行 SQL 查询子任务。"""
        from src.schema_agent import (
            _make_search_databases,
            _make_search_fields,
            _make_get_table_schema,
            _make_list_databases,
            _make_submit_sql,
        )

        captured = {"sql_list": [], "last_db": ""}
        sub_agent = create_react_agent(
            model=self.llm,
            tools=[
                _make_search_databases(self.store, self.model),
                _make_search_fields(self.store, self.model),
                _make_get_table_schema(),
                _make_list_databases(),
                _make_submit_sql(captured),
            ],
            prompt=SUB_TASK_PROMPT.format(prompt=prompt, context=context),
        )

        result = sub_agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": 15},
        )

        # 提取最终 AI 回答
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                return msg.content.strip()

        return "（子任务未能生成结果）"

    def _run_synthesize(self, prompt: str, context: str) -> str:
        """综合前面任务的结果，生成文字结论。"""
        full_prompt = (
            f"根据以下查询结果，回答用户的问题。\n\n"
            f"用户问题: {prompt}\n\n"
            f"已有数据:\n{context}\n\n"
            f"请用中文给出简洁的综合结论。"
        )
        resp = self.llm.invoke(full_prompt, temperature=0.1, max_tokens=500)
        return resp.content.strip()

    def _run_chart_task(self, prompt: str, context: str) -> str:
        """从上下文提取数据并生成图表。"""
        # 先让 LLM 从上下文中提取数据和图表参数
        extract_prompt = (
            f"从以下查询结果中提取图表所需的数据和参数。\n\n"
            f"图表需求: {prompt}\n\n"
            f"数据:\n{context}\n\n"
            f"返回 JSON: {{\"chart_type\": \"bar|pie|line\", \"data_text\": \"从context中复制原始数据文本\", \"title\": \"图表标题\"}}"
        )
        resp = self.llm.invoke(extract_prompt, temperature=0.1, max_tokens=300)
        raw = resp.content.strip()

        try:
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            params = json.loads(raw)
            from src.chart_tool import create_chart
            return create_chart(
                chart_type=params.get("chart_type", "bar"),
                data_text=params.get("data_text", context),
                title=params.get("title", prompt),
            )
        except (json.JSONDecodeError, Exception):
            # fallback: 直接用 context 画柱状图
            from src.chart_tool import create_chart
            return create_chart("bar", context, title=prompt[:50])

    # ---- 聚合所有结果 ------------------------------------------------------

    def get_task_stats(self, tasks: list[dict]) -> dict:
        """统计任务执行状态。"""
        success, failed = 0, 0
        for t in tasks:
            r = self._results_cache.get(t["id"])
            if isinstance(r, dict) and r.get("success"):
                success += 1
            else:
                failed += 1
        return {"total": len(tasks), "success": success, "failed": failed}

    def aggregate(self, tasks: list[dict]) -> str:
        """综合所有子任务结果（含失败容错），生成最终答案。"""
        parts = []
        for t in tasks:
            tid = t["id"]
            if tid in self._results_cache:
                r = self._results_cache[tid]
                if isinstance(r, dict):
                    status = "成功" if r.get("success") else "失败"
                    result = r.get("result", "")
                else:
                    status, result = "成功", str(r)
                parts.append(
                    f"### 子任务 {tid} (状态: {status}): {t.get('prompt', '')}\n{result}"
                )

        if not parts:
            return "（无任务结果）"

        stats = self.get_task_stats(tasks)
        combined = "\n\n".join(parts)

        # LLM 综合（告知部分任务可能失败）
        agg_prompt = (
            f"综合以下子任务的执行结果（共 {stats['total']} 个任务，"
            f"成功 {stats['success']} 个，失败 {stats['failed']} 个），"
            f"用中文给出一份完整的回答。失败的任务说明无法获取该部分数据，请基于已有数据回答。\n\n{combined}"
        )
        resp = self.llm.invoke(agg_prompt, temperature=0.1, max_tokens=1000)
        return resp.content.strip()

    def reset(self):
        """清空结果缓存。"""
        self._results_cache = {}


# ---- TaskBus: asyncio 事件总线并行执行引擎 ---------------------------------

class TaskBus:
    """asyncio 事件总线：依赖满足 → 发布到总线 → 异步并行执行。

    特性:
    - asyncio 协程并发，LLM 调用通过 to_thread 非阻塞
    - 事件驱动：任务完成 → bus.put() → 只通知直接下游（O(1)）
    - 预计算下游订阅表，避免 O(n²) 全量扫描
    - 失败自动重试（最多 2 次）+ 降级容错
    - asyncio.wait_for 超时保护（默认 120s）
    """

    MAX_RETRIES = 2
    TASK_TIMEOUT = 120    # 单任务超时（秒）
    BUS_TIMEOUT = 300     # 总线空等超时（秒）

    def __init__(self, manager: MultiTaskManager):
        self.manager = manager
        self.tasks: list[dict] = []
        self.retry_count: dict[str, int] = {}
        self.downstream_map: dict[str, list[str]] = {}   # task_id → [直接下游 task_id]
        self.pending_deps: dict[str, int] = {}           # task_id → 未完成的依赖数
        self.running: set[str] = set()

    # ---- 公共 API ----------------------------------------------------------

    def execute_all(self, tasks: list[dict]) -> dict:
        """同步入口：内部调用 asyncio.run()。"""
        import asyncio
        return asyncio.run(self._async_execute_all(tasks))

    async def _async_execute_all(self, tasks: list[dict]) -> dict:
        """异步执行全部任务。"""
        import asyncio
        import logging
        log = logging.getLogger("TaskBus")

        self.tasks = tasks
        self.retry_count = {t["id"]: 0 for t in tasks}

        # 预计算：下游订阅表 + 每个任务的待完成依赖计数
        self._build_dependency_graph(tasks)

        log.info("[TaskBus] 启动: %d 个任务, max retries=%d, timeout=%ds (asyncio)",
                 len(tasks), self.MAX_RETRIES, self.TASK_TIMEOUT)

        # 事件总线
        bus: asyncio.Queue = asyncio.Queue()

        # 第一轮：发布所有无依赖任务
        for task in tasks:
            if self.pending_deps[task["id"]] == 0:
                asyncio.create_task(self._run_task(task, bus, log))

        completed = 0
        while completed < len(tasks):
            try:
                msg = await asyncio.wait_for(bus.get(), timeout=self.BUS_TIMEOUT)
            except asyncio.TimeoutError:
                log.error("[TaskBus] 总线超时 (%ds)，强制终止", self.BUS_TIMEOUT)
                break

            task_id = msg["task_id"]
            status = msg["status"]
            completed += 1

            if status == "done":
                log.info("[TaskBus]   %s: OK", task_id)
                self._notify_downstream(task_id, bus, log)
            elif status == "retry":
                log.warning("[TaskBus]   %s: 重试 %d/%d",
                            task_id, self.retry_count[task_id], self.MAX_RETRIES)
                asyncio.create_task(self._run_task(
                    self._find_task(task_id), bus, log
                ))
                completed -= 1  # 重试不算完成
            elif status == "degraded":
                log.error("[TaskBus]   %s: 降级 (已重试 %d 次)", task_id, self.MAX_RETRIES)
                self._notify_downstream(task_id, bus, log)

        stats = self.manager.get_task_stats(tasks)
        log.info("[TaskBus] 完成: %d 成功, %d 失败",
                 stats["success"], stats["failed"])
        return stats

    # ---- 依赖图构建 --------------------------------------------------------

    def _build_dependency_graph(self, tasks: list[dict]):
        """预计算下游订阅表和依赖计数（O(n) 一次性）。"""
        task_map = {t["id"]: t for t in tasks}

        for task in tasks:
            tid = task["id"]
            self.pending_deps[tid] = len(task.get("depends", []))
            self.downstream_map[tid] = []

        # 构建反向索引：被依赖者 → 依赖它的下游任务
        for task in tasks:
            for dep_id in task.get("depends", []):
                if dep_id in self.downstream_map:
                    self.downstream_map[dep_id].append(task["id"])

    # ---- 事件通知 ----------------------------------------------------------

    def _notify_downstream(self, finished_id: str, bus, log):
        """任务完成后只通知直接下游（O(1) 而非 O(n²)）。"""
        for downstream_id in self.downstream_map.get(finished_id, []):
            self.pending_deps[downstream_id] -= 1
            if self.pending_deps[downstream_id] == 0:
                task = self._find_task(downstream_id)
                if task:
                    log.info("[TaskBus] → 依赖满足，触发 %s", downstream_id)
                    asyncio.create_task(self._run_task(task, bus, log))

    # ---- 任务执行 ----------------------------------------------------------

    async def _run_task(self, task: dict, bus, log):
        """异步执行单个任务（同步 LLM 调用通过 to_thread 非阻塞）。"""
        import asyncio
        task_id = task["id"]
        self.running.add(task_id)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.manager.execute_single_task, task),
                timeout=self.TASK_TIMEOUT,
            )

            if result["success"]:
                await bus.put({"task_id": task_id, "status": "done"})
            else:
                self._handle_failure(task, bus)

        except asyncio.TimeoutError:
            log.error("[TaskBus]   %s: 超时 (%ds)", task_id, self.TASK_TIMEOUT)
            # 超时视为失败，走重试/降级
            self.manager._results_cache[task_id] = {
                "success": False,
                "result": f"任务执行超时 ({self.TASK_TIMEOUT}s)",
            }
            self._handle_failure(task, bus)

        except Exception as e:
            log.error("[TaskBus]   %s: 异常: %s", task_id, str(e)[:100])
            self.manager._results_cache[task_id] = {
                "success": False,
                "result": f"任务异常: {str(e)[:200]}",
            }
            self._handle_failure(task, bus)

        finally:
            self.running.discard(task_id)

    def _handle_failure(self, task: dict, bus):
        """失败处理：重试或降级。"""
        task_id = task["id"]
        current = self.retry_count.get(task_id, 0)

        if current < self.MAX_RETRIES:
            self.retry_count[task_id] = current + 1
            bus.put_nowait({"task_id": task_id, "status": "retry"})
        else:
            bus.put_nowait({"task_id": task_id, "status": "degraded"})

    def _find_task(self, task_id: str) -> dict | None:
        """按 ID 查找任务定义。"""
        for t in self.tasks:
            if t["id"] == task_id:
                return t
        return None


# ---- 命令行测试入口 --------------------------------------------------------

if __name__ == "__main__":
    import os, sys, time, logging
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ["HF_HUB_OFFLINE"] = "1"

    from langchain_openai import ChatOpenAI
    from src.field_embedder import FieldEmbedder
    from src.qdrant_store import QdrantStore

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("MultiTask")

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("请设置 DEEPSEEK_API_KEY")
        exit(1)

    llm = ChatOpenAI(model="deepseek-chat", api_key=api_key,
                     base_url="https://api.deepseek.com/v1", temperature=0.1, timeout=30)
    embedder = FieldEmbedder(offline=True)
    store = QdrantStore(timeout=5)

    mgr = MultiTaskManager(llm, store, embedder.model)

    # 测试查询
    query = "查concert_singer和singer数据库分别有多少歌手，然后对比"
    print(f"查询: {query}\n")

    # 1. 分解
    log.info("=== 1. 任务分解 ===")
    tasks = mgr.decompose(query)
    for t in tasks:
        print(f"  {t['id']}: [{t['type']}] {t['prompt'][:60]}  depends={t.get('depends',[])}")

    # 2. 依赖关系
    log.info("\n=== 2. 依赖关系 ===")
    for t in tasks:
        deps = t.get("depends", [])
        dep_str = f" ← {', '.join(deps)}" if deps else ""
        print(f"  {t['id']}{dep_str}: [{t['type']}] {t['prompt'][:60]}")

    # 3. TaskBus 事件驱动并行执行
    log.info("\n=== 3. TaskBus 并行执行 ===")
    bus = TaskBus(mgr)
    t0 = time.time()
    stats = bus.execute_all(tasks)
    print(f"\n  总耗时: {time.time()-t0:.1f}s")
    print(f"  成功: {stats['success']}, 失败: {stats['failed']}")

    # 4. 聚合
    log.info("\n=== 4. 最终聚合 ===")
    final = mgr.aggregate(tasks)
    print(final)
