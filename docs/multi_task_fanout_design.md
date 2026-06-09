# Multi-Task Fan-out 并行执行方案

## 1. 核心问题

用户查询可能包含多个独立子任务，这些任务之间存在依赖关系：
- 无依赖 → 可以**并行执行**
- 有依赖 → 必须**串行执行**（等前面的任务完成）

例如："查 concert_singer 和 singer 的歌手数，然后对比"

```
A: 查concert_singer歌手数 ─┐
B: 查singer歌手数         ─┤ 无依赖 → 并行 (Wave 1)
                            ↓
C: 对比A和B的结果 ──────────── 依赖A,B → 串行 (Wave 2)
```

## 2. 架构概览

```
用户 Query
    ↓
Rewrite Node (查询规范化)
    ↓
Router (新增 "multi" 意图)
    ↓
Decomposer Node (LLM 分析 → 输出任务 DAG JSON)
    ↓
Scheduler Node (依赖分析 → 分组为 Wave)
    ↓
┌─ Wave 1: [Task A] ∥ [Task B] ─ 并行 (Send fan-out) ─┐
│                                                       │
├─ Wave 2: [Task C] ─ 串行 (等 Wave 1 完成) ──────────┤
│                                                       │
└─ Aggregator Node (LLM 综合所有结果 → 最终答案) ──────┘
```

## 3. 任务 DAG 格式

Decomposer 输出的 JSON 结构：

```json
{
  "tasks": [
    {
      "id": "A",
      "type": "sql_query",
      "prompt": "查询concert_singer数据库的歌手数量",
      "depends": []
    },
    {
      "id": "B",
      "type": "sql_query",
      "prompt": "查询singer数据库的歌手数量",
      "depends": []
    },
    {
      "id": "C",
      "type": "synthesize",
      "prompt": "对比A和B的结果，判断哪个数据库歌手更多",
      "depends": ["A", "B"]
    }
  ]
}
```

### 任务类型

| type | 描述 | 示例 |
|------|------|------|
| `sql_query` | 数据库查询 | 查某库某表的数据 |
| `synthesize` | 综合对比 | 对比两个查询结果 |
| `chart` | 生成图表 | 根据数据画柱状图/饼图 |

## 4. 调度算法 (Wave Scheduler)

```python
def build_execution_plan(tasks):
    completed = set()
    remaining = list(tasks)
    waves = []

    while remaining:
        wave = []
        still_waiting = []
        for task in remaining:
            if all(dep in completed for dep in task["depends"]):
                wave.append(task)        # 依赖已满足 → 加入当前波次
            else:
                still_waiting.append(task)  # 依赖未满足 → 等待下一波

        if not wave:
            break  # 死锁检测：无任务可执行

        waves.append(wave)
        for task in wave:
            completed.add(task["id"])
        remaining = still_waiting

    return waves
```

## 5. Fan-out 实现 (LangGraph Send)

```python
from langgraph.types import Send

def schedule_waves(state):
    tasks = state["pending_tasks"]
    ready = [t for t in tasks
             if all(dep in state["task_results"] for dep in t["depends"])]

    if not ready:
        return END  # 全部完成

    # 并行发送到 task_executor 节点
    return [Send("task_executor", {"current_task": t}) for t in ready]
```

## 6. 子任务执行器

每个 `task_executor` 实例是独立的 ReAct Agent，根据 `task["type"]` 选择执行模式：

- **sql_query**: 使用 5 个工具 (search_databases → search_fields → get_table_schema → submit_sql)
- **synthesize**: 纯 LLM 推理，读取依赖任务的结果，输出文字结论
- **chart**: LLM 从上下文提取数据 + 调用 `create_chart` 工具

## 7. 依赖数据传递

子任务之间通过 `_results_cache` 共享数据：

```python
class MultiTaskManager:
    _results_cache: dict[str, str] = {}  # {task_id: result_text}

    def execute_single_task(self, task):
        context = ""
        for dep_id in task["depends"]:
            if dep_id in self._results_cache:
                context += f"任务{dep_id}的结果:\n{self._results_cache[dep_id]}"

        # 将 context 注入到子 Agent 的 Prompt 中
        ...
```

## 8. Router 集成

在 `schema_agent.py` 的 ROUTER_PROMPT 中新增一条规则，将多任务查询路由到 `multi` 路径：

```
- 用户消息包含"并且"、"同时"、"分别...然后对比"、"先...再..."等词
  → {"intent": "multi"}
```

## 9. 已实现文件

| 文件 | 内容 |
|------|------|
| `src/multi_task_agent.py` | MultiTaskManager：分解/调度/执行/聚合 |
| `src/schema_agent.py` | 待集成：Router 增加 `multi` 意图 + Graph 增加节点 |

## 10. 验证结果

测试查询: "查concert_singer和singer数据库分别有多少歌手，然后对比"

```
=== 任务分解 ===
A: [sql_query] 查询concert_singer数据库的歌手数量  depends=[]
B: [sql_query] 查询singer数据库的歌手数量         depends=[]
C: [synthesize] 对比A和B的查询结果，给出结论      depends=['A','B']

=== 执行计划 ===
Wave 1: ∥ A  B   (无依赖 → 并行)
Wave 2: → C       (依赖A,B → 串行)
```
