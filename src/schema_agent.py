"""
Multi-Agent Text-to-SQL: LangGraph StateGraph 架构

架构:
    User Query
        ↓
    Router Node  ──chat──→ Chat Node ──→ END
        │
       sql
        ↓
    SQL Generator Node (ReAct, search_fields + get_table_schema)
        ↓
    Safety Check Node (规则引擎)
        ├── fail → 回到 SQL Generator (fix)
        └── pass ↓
    Execute Node (sqlite3)
        ├── success → 返回结果
        └── failure → 回到 SQL Generator (fix, max 3 次)

用法:
    agent = SchemaAgent()
    result = agent.run("统计每位歌手举办的演唱会数量")
"""

import os
import sys
import sqlite3
import logging
import time
import re
from typing import Annotated, Literal
from typing_extensions import TypedDict

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, SystemMessage,
)
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

# ---- 配置 ------------------------------------------------------------------

SPIDER_DATA = os.path.join(os.path.dirname(__file__), "..", "spider_data", "database")

LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

_logger = logging.getLogger("MultiAgent")
_logger.setLevel(logging.DEBUG)
_console = logging.StreamHandler()
_console.setLevel(logging.DEBUG)
_console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
_logger.addHandler(_console)
log = _logger


# ---- State 定义 ------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    db_name: str
    generated_sql: str
    final_answer: str
    rewritten_query: str              # rewrite 后的规范化查询
    insight_report: str               # data_insight 趋势分析报告
    running_summary: str              # LLM 滚动摘要（长期压缩记忆）


# ---- 工具工厂 ---------------------------------------------------------------

def _make_search_fields(store, model):
    @tool
    def search_fields(query_text: str, db_name: str = "") -> str:
        """在 Qdrant 中搜索与自然语言描述最相关的数据库字段。
        参数 query_text: 中文关键词，如 "歌手姓名"
        参数 db_name: 可选，限定数据库名。支持逗号分隔多个库名，如 "hr_1,employee_hire_evaluation"
                      留空则搜索全部数据库"""
        if db_name:
            db_list = [d.strip() for d in db_name.split(",") if d.strip()]
            filter_dict = {"database": db_list if len(db_list) > 1 else db_list[0]}
        else:
            filter_dict = None
        hits = store.search(query_text, model, top_k=10, filter_dict=filter_dict)
        if not hits:
            return "未找到匹配的字段。"
        lines = []
        for i, h in enumerate(hits):
            p = h["payload"]
            tags = []
            if p.get("is_pk"): tags.append("主键")
            if p.get("is_fk"): tags.append(
                f"外键→{p.get('ref_table','')}.{p.get('ref_column','')}"
            )
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(
                f"{i+1}. {p['field_id']} (类型={p['type']}){tag_str}\n"
                f"   描述: {p.get('description', '-')}"
            )
        return "\n".join(lines)
    return search_fields


def _make_get_table_schema():
    @tool
    def get_table_schema(db_name: str, table_name: str) -> str:
        """获取某数据库中指定表的完整建表语句（DDL）"""
        sqlite_path = os.path.join(SPIDER_DATA, db_name, f"{db_name}.sqlite")
        if not os.path.exists(sqlite_path):
            return f"数据库 {db_name} 不存在"
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        conn.close()
        return schema[0] if (schema and schema[0]) else f"表 {table_name} 不存在"
    return get_table_schema


def _make_list_databases():
    @tool
    def list_databases() -> str:
        """列出当前所有可用的数据库名称"""
        if not os.path.isdir(SPIDER_DATA):
            return "数据库目录不可用"
        dbs = sorted(d for d in os.listdir(SPIDER_DATA)
                     if os.path.isdir(os.path.join(SPIDER_DATA, d)) and not d.startswith("."))
        return "\n".join(dbs)
    return list_databases


def _make_search_databases(store, model):
    @tool
    def search_databases(query_text: str) -> str:
        """根据用户问题语义搜索最相关的数据库。必须先调用此工具确定数据库，
        再用 search_fields 的 db_name 参数限定在该数据库中搜索字段。

        参数 query_text: 用户问题的关键词，如 "员工 职位 姓名" """
        hits = store.search(query_text, model, top_k=15, filter_dict=None)
        if not hits:
            return "未找到相关数据库。"

        # 按数据库聚合分数
        db_scores = {}
        for h in hits:
            db = h["payload"]["database"]
            if db not in db_scores:
                db_scores[db] = {"total_score": 0, "count": 0, "top_fields": []}
            db_scores[db]["total_score"] += h["score"]
            db_scores[db]["count"] += 1
            if len(db_scores[db]["top_fields"]) < 3:
                db_scores[db]["top_fields"].append(
                    f"{h['payload']['table']}.{h['payload']['column']}"
                )

        # 按总分数排序
        ranked = sorted(db_scores.items(),
                       key=lambda x: x[1]["total_score"], reverse=True)

        lines = ["最相关的数据库 (按相关度排序):"]
        for i, (db, info) in enumerate(ranked[:5]):
            fields_preview = ", ".join(info["top_fields"])
            lines.append(
                f"{i+1}. {db} (相关度: {info['total_score']:.2f}, "
                f"命中字段数: {info['count']}, "
                f"示例字段: {fields_preview})"
            )
        return "\n".join(lines)
    return search_databases


def _make_create_chart():
    """create_chart 工具: 根据查询结果数据绘制图表。"""
    @tool
    def create_chart(chart_type: str, data_text: str, title: str = "",
                     x_label: str = "", y_label: str = "") -> str:
        """根据 SQL 查询结果数据生成图表（饼状图/柱状图/折线图）。

        参数 chart_type: 图表类型: "pie"(饼图) / "bar"(柱状图) / "line"(折线图)
        参数 data_text: submit_sql 返回的完整数据文本（直接复制粘贴）
        参数 title: 图表标题，如 "部门薪资分布"
        参数 x_label: X轴标签（饼图可省略）
        参数 y_label: Y轴标签（饼图可省略）

        返回: 生成的图表文件路径"""
        from src.chart_tool import create_chart as _create_chart
        return _create_chart(chart_type, data_text, title, x_label, y_label)
    return create_chart


def _make_submit_sql(captured):
    """submit_sql 工具: 安全检查 + 执行 + 返回结果。
    Agent 可以多次调用此工具来查询多个数据库，累积结果后给出最终回答。"""
    @tool
    def submit_sql(sql: str, db_name: str) -> str:
        """执行一条 SQL 查询并返回结果。包含自动安全检查。
        参数 sql: 完整的 SQL SELECT 语句
        参数 db_name: 目标数据库名称
        可以多次调用以查询不同数据库或不同表的数据。"""
        from src.safety_checker import SafetyChecker

        # 1. 安全检查
        safety = SafetyChecker.check(sql)
        if not safety["passed"]:
            errors = "; ".join(safety["errors"])
            return f"安全检查失败: {errors}\n请修正 SQL 后重试。"

        # 2. 执行 SQL
        sqlite_path = os.path.join(SPIDER_DATA, db_name, f"{db_name}.sqlite")
        if not os.path.exists(sqlite_path):
            return f"数据库 {db_name} 不存在"

        try:
            conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(sql)

            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(50)
            conn.close()

            # 记录执行的 SQL
            prev = captured.get("sql_list", [])
            prev.append({"sql": sql, "db_name": db_name, "rows": len(rows)})
            captured["sql_list"] = prev
            captured["last_db"] = db_name

            if not rows:
                return "(查询成功，但返回了 0 行数据)\n请考虑是否需要调整查询条件或切换到另一个数据库。"

            lines = [" | ".join(columns), "-" * 50]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
            if len(rows) == 50:
                lines.append("... (结果被截断到 50 行)")

            return f"({len(rows)} 行)\n" + "\n".join(lines)
        except Exception as e:
            return f"SQL 执行错误: {e}\n请分析错误原因并修正 SQL。"
    return submit_sql


# ---- Agent 类 --------------------------------------------------------------

class SchemaAgent:
    """Multi-Agent Text-to-SQL，基于 LangGraph StateGraph。支持多轮对话。

    用法:
        agent = SchemaAgent()
        result = agent.run("统计每位歌手举办的演唱会数量")
        result = agent.run("那这些歌手都是哪个国家的？")  # 上下文感知

        # 交互式模式
        agent.chat_loop()
    """

    MAX_CONTEXT_TOKENS = 6000   # Prompt 中历史上下文的最大 token 数
    RESERVE_TOKENS = 2000       # 保留给 System Prompt + 工具返回 + 当前 query
    MAX_HISTORY_MSG = 40        # 硬上限兜底（防止异常情况消息数爆炸）

    def __init__(self, api_key=None, model="deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "未设置 DeepSeek API Key。请设置环境变量 DEEPSEEK_API_KEY"
            )

        # ---- LLM ----
        self.llm = ChatOpenAI(
            model=model,
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.1,
            timeout=30,
            max_retries=2,
        )

        # ---- Embedding + Qdrant ----
        os.environ["HF_HUB_OFFLINE"] = "1"
        log.info("加载 embedding 模型 (首次导入约需 15s)...")
        from src.field_embedder import FieldEmbedder
        from src.qdrant_store import QdrantStore
        self.embedder = FieldEmbedder(offline=True)
        self.store = QdrantStore(timeout=5)
        log.info("embedding 模型加载完成")

        # Qdrant 健康检查
        try:
            self.store.collection_info("schema_fields")
            log.info("Qdrant 连接正常")
            self._init_memory()
        except Exception as e:
            log.warning("Qdrant 连接失败: %s。检索功能将不可用。请启动 Qdrant: docker start qdrant", e)

        # ---- SQL Generator 子 Agent ----
        gen_tools = [
            _make_search_databases(self.store, self.embedder.model),
            _make_search_fields(self.store, self.embedder.model),
            _make_get_table_schema(),
            _make_list_databases(),
        ]
        self._sql_generator = create_react_agent(
            model=self.llm,
            tools=gen_tools,
            prompt=self._SQL_GEN_PROMPT,
        )

        # ---- StateGraph ----
        self.graph = self._build_graph()

        # ---- 多轮对话状态 ----
        self.reset()

    # ---- System Prompts ----------------------------------------------------

    ROUTER_PROMPT = """根据对话历史和用户最新消息判断意图，仅返回 JSON。

判断标准:
- 用户消息涉及数据库查询、数据统计、表查询、某类数据的数量/排名/列表等 → {"intent": "sql"}
- 用户消息是普通对话、闲聊、问好、追问已返回结果的含义 → {"intent": "chat"}
- 用户消息包含"这些"、"那个"、"上面"等指代词，且指向之前的 SQL 查询结果 → {"intent": "sql"}
- 用户消息要求对已有查询结果做进一步分析/过滤/排序 → {"intent": "sql"}
- 用户消息要求分析数据趋势、数据分布、数据特征、数据概况、数据洞察 → {"intent": "insight"}
- 用户消息包含"分析趋势"、"分布"、"画像"、"洞察"、"总体情况"、"数据特点"等词 → {"intent": "insight"}
- 用户消息要求"画图"、"制图"、"饼图"、"柱状图"、"折线图"、"可视化" → 根据是否含"分析"/"趋势"判断为insight或sql
- 用户消息包含多个独立查询或"并列"、"同时"、"分别...然后对比"、"以及"、"并且"连接两个以上查询 → {"intent": "multi"}

只返回 JSON，不要其他内容。"""

    _SQL_GEN_PROMPT = """你是一个 Text-to-SQL 专家。根据用户的问题和对话上下文，按照以下流程操作：

## 工作流程
**第 1 步**: 调用 search_databases(query_text) 确定最相关的数据库
**第 2 步**: 调用 search_fields(query_text, db_name) 搜索字段（可用逗号分隔多个库名）
**第 3 步**: 如需查看完整表结构，调用 get_table_schema(db_name, table_name)
**第 4 步**: 生成 SQL SELECT 语句，调用 submit_sql(sql, db_name) 执行查询
**第 5 步**: 根据返回结果判断是否需要更多查询。如果需要其他数据库的数据，回到第 1 步用另一个数据库继续查询
**第 6 步**: 综合所有查询结果，给出完整的中文回答

## 重要规则
- submit_sql 可以多次调用，用来查询不同数据库或不同表的数据
- 对于对比类问题（"A库和B库分别有多少"），分别对每个库执行查询，然后汇总对比
- 如果查询返回 0 行，尝试调整条件或确认数据库是否正确
- SQLite 中表名和字段名用双引号包裹
- 多表查询通过外键关联正确 JOIN
- SQL 不要包含注释
- 如果用户要求画图/可视化，在查询到数据后调用 create_chart 工具生成图表(pie/bar/line)
- 当获得足够信息回答用户问题后，直接给出最终答案（无需调用工具）"""

    CHAT_PROMPT = """你是一个友好的 AI 助手。请用中文回复用户的问题。
如果用户问你能做什么，告诉用户你可以帮助查询和分析数据库中的数据。
如果用户追问之前的查询结果，结合上下文给出有帮助的回答。"""

    INSIGHT_PROMPT = """你是一个数据分析师。用户想了解数据库中的数据分布和趋势。

## 工作流程
**第 1 步**: 调用 search_databases(query_text) 定位目标数据库
**第 2 步**: 调用 get_table_schema(db_name, table_name) 了解表结构，关注数值列、日期列、分类列
**第 3 步**: 针对数据从以下维度执行探索性查询:
   a. **数据概览**: COUNT(*)，了解数据量
   b. **分布分析**: 数值列的 MIN/MAX/AVG/COUNT/SUM
   c. **分类分布**: 分类列 GROUP BY + COUNT，按数量降序
   d. **时间趋势**: 如果存在日期列，按年/月 GROUP BY + 聚合
   e. **Top-N**: 找出排名前 5 的实体（如销量最高的产品）
   f. **占比分析**: 各部分占整体的百分比
**第 4 步**: 调用 submit_sql(sql, db_name) 执行每条查询
**第 5 步**: 综合所有查询结果，输出一份数据洞察报告

## 重要规则
- 至少覆盖 3 个分析维度（数据概览 + 分布 + 分类/时间）
- 每条 SQL 执行后根据结果决定下一步探索方向
- 发现有趣的数据模式（如极值、集中度高）时深入挖掘
- **可视化**: 对于分组/对比类数据，调用 create_chart 生成图表（饼图展示占比、柱状图展示对比、折线图展示趋势）
- 最终报告格式: 先用 2-3 句话总结关键发现，再分点列出数据支撑
- 如果某维度无数据可分析（如无日期列），跳过该维度
- submit_sql 和 create_chart 可以多次调用
- SQL 不要包含注释"""

    REWRITE_PROMPT = """你是一个查询改写助手。将用户的原始查询改写为规范化的数据库查询语句。

## 改写规则
1. **去噪**：移除口语化填充词（"帮我"、"那个"、"我想查一下"等）和无关标点。**保留**"画图"、"饼图"、"柱状图"、"折线图"、"可视化"等制图相关指令
2. **标准化**：
   - 单位统一：将所有中文数字统一为阿拉伯数字（"三十"→30，"一百多"→"100以上"）
   - 时间统一：将口语化时间转为标准表述（"去年"→具体年份，"上个月"→具体月份，"圣诞季"→"12月"）
   - 标点统一：使用中文标点
3. **指代消解**：根据对话历史，将"这些"、"那个"、"上面"等指代词替换为具体实体
   - 如上一轮查了"Justin Brown等歌手"，本轮"这些歌手的国家"→"Justin Brown等歌手的国家"
4. **补全语义**：如果查询省略了表名或条件，根据上下文补充
5. **截断**：如果原查询包含大段无关文字，只保留核心查询意图

## 输出格式
只返回改写后的一句话查询，不要解释，不要额外文字。保留查询的原始意图完整性。

## 对话历史
{history}

## 当前查询
{query}

## 改写结果"""

    # ---- StateGraph 构建 ---------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("rewrite", self._rewrite_node)
        builder.add_node("router", self._router_node)
        builder.add_node("chat", self._chat_node)
        builder.add_node("sql_generator", self._sql_generator_node)
        builder.add_node("insight_generator", self._insight_generator_node)
        builder.add_node("multi_task", self._multi_task_node)

        # Rewrite 在最前面，所有请求统一经过规范化
        builder.set_entry_point("rewrite")
        builder.add_edge("rewrite", "router")

        builder.add_conditional_edges(
            "router", self._route_after_router,
            {"chat": "chat", "sql": "sql_generator",
             "insight": "insight_generator", "multi": "multi_task"},
        )
        builder.add_edge("chat", END)
        builder.add_edge("sql_generator", END)
        builder.add_edge("insight_generator", END)
        builder.add_edge("multi_task", END)

        return builder.compile()

    # ---- Router Node -------------------------------------------------------

    def _router_node(self, state: AgentState) -> dict:
        # 优先使用 Rewrite 后的规范化查询做意图分类
        query = state.get("rewritten_query", "")
        if not query:
            last_msg = state["messages"][-1]
            query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        log.info("[Router] 分析意图: %s", query[:80])

        router_msgs = [SystemMessage(content=self.ROUTER_PROMPT)]
        clean_history = self._strip_tool_messages(state["messages"])
        history = clean_history[-4:]
        if history:
            context = "\n".join(
                f"[{'用户' if isinstance(m, HumanMessage) else 'AI'}]: {m.content[:100]}"
                for m in history[-4:]
            )
            router_msgs.append(HumanMessage(
                content=f"对话历史:\n{context}\n\n当前问题: {query}"
            ))
        else:
            router_msgs.append(HumanMessage(content=query))

        resp = self.llm.invoke(router_msgs)

        intent = "chat"
        try:
            import json
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].split("```")[0]
            data = json.loads(raw)
            intent = data.get("intent", "chat")
        except Exception:
            # fallback: 含数据库关键词 → sql/insight/multi
            multi_keywords = ["分别", "同时", "以及", "对比", "比较"]
            insight_keywords = ["分析", "趋势", "分布", "画像", "洞察", "特点", "概况"]
            sql_keywords = ["查询", "统计", "数量", "数据", "表", "字段", "SQL", "数据库"]
            if any(kw in query for kw in multi_keywords) and \
               any(kw in query for kw in sql_keywords):
                intent = "multi"
            elif any(kw in query for kw in insight_keywords):
                intent = "insight"
            elif any(kw in query for kw in sql_keywords):
                intent = "sql"

        log.info("[Router] 意图: %s", intent)
        return {"intent": intent}

    # ---- Rewrite Node ------------------------------------------------------

    def _rewrite_node(self, state: AgentState) -> dict:
        """查询改写节点：去噪、标准化、指代消解、补全语义。"""
        # 获取当前用户查询
        query = ""
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                query = m.content
                break

        # 构建对话历史
        history_parts = []
        prev_ai = None
        for m in state["messages"]:
            if isinstance(m, HumanMessage) and m.content != query:
                history_parts.append(f"用户: {m.content}")
            elif isinstance(m, AIMessage) and m.content:
                prev_ai = m.content[:200]
        if prev_ai:
            history_parts.append(f"AI: {prev_ai}")
        history = "\n".join(history_parts[-6:]) if history_parts else "（无历史）"

        # 注入长期记忆
        memory = self._retrieve_memory(query)
        summary = state.get("running_summary", "")
        extra = ""
        if memory: extra += f"\n{memory}"
        if summary: extra += f"\n[历史摘要]: {summary[:300]}"

        prompt = self.REWRITE_PROMPT.format(history=history + extra, query=query)
        log.info("[Rewrite] 原始: %s", query[:80])

        try:
            resp = self.llm.invoke(prompt, temperature=0.1, max_tokens=200)
            rewritten = resp.content.strip()
            # 清理 LLM 可能的冗余输出
            if rewritten.startswith("改写结果") or rewritten.startswith("改写"):
                rewritten = rewritten.split("\n", 1)[-1].strip()
            rewritten = rewritten.strip('"').strip("'").strip()
        except Exception:
            rewritten = query

        # 如果改写后为空或变化太大（明显错误），回退到原始
        if not rewritten or len(rewritten) < 3:
            rewritten = query

        log.info("[Rewrite] 改写: %s", rewritten[:80])

        return {
            "rewritten_query": rewritten,
            # 将改写后的查询作为新的 HumanMessage 追加，供 SQL Generator 使用
            "messages": [HumanMessage(content=f"[规范化查询] {rewritten}")],
        }

    # ---- Chat Node ---------------------------------------------------------

    def _chat_node(self, state: AgentState) -> dict:
        clean_msgs = self._strip_tool_messages(state["messages"])
        # 用改写后的查询替换原始消息，提升回复质量
        rewritten = state.get("rewritten_query", "")
        if rewritten:
            clean_msgs = [
                HumanMessage(content=rewritten)
                if (isinstance(m, HumanMessage) and m.content == state["messages"][-1].content)
                else m
                for m in clean_msgs
            ]
        msgs = [SystemMessage(content=self.CHAT_PROMPT)] + clean_msgs
        resp = self.llm.invoke(msgs)
        log.info("[Chat] %s", resp.content[:100])
        return {"final_answer": resp.content, "messages": [resp]}

    # ---- SQL Generator Node ------------------------------------------------

    def _sql_generator_node(self, state: AgentState) -> dict:
        # 优先使用 rewrite 后的规范化查询
        user_query = state.get("rewritten_query", "")
        if not user_query:
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    # 跳过 Rewrite 节点追加的标记消息
                    content = m.content
                    if content.startswith("[规范化查询]"):
                        user_query = content.replace("[规范化查询] ", "")
                    else:
                        user_query = content
                    break

        log.info("[SQL Gen] %s", user_query[:80])

        # 挂载所有工具（submit_sql 内置安全检查 + 执行）
        captured = {"sql_list": [], "last_db": ""}
        gen_with_submit = create_react_agent(
            model=self.llm,
            tools=[
                _make_search_databases(self.store, self.embedder.model),
                _make_search_fields(self.store, self.embedder.model),
                _make_get_table_schema(),
                _make_list_databases(),
                _make_submit_sql(captured),
                _make_create_chart(),
            ],
            prompt=self._SQL_GEN_PROMPT,
        )

        # 构建带上下文的子 Agent 输入
        gen_messages = []
        history_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if len(history_msgs) > 1:
            prev = history_msgs[-2]
            gen_messages.append(HumanMessage(
                content=f"(上一轮查询: {prev.content[:150]})"
            ))
        gen_messages.append(HumanMessage(content=user_query))

        # 使用 callback 记录 ReAct 循环的每一步
        step_counter = [0]  # 用列表实现闭包可变引用

        class StepLoggerCallback(BaseCallbackHandler):
            """LangChain 回调：记录 LLM 推理和工具调用的每一步。"""
            def on_tool_start(self, serialized, input_str, **kwargs):
                step_counter[0] += 1
                inp = str(input_str)[:150].replace("\n", " ")
                log.info("[SQL Gen]   Step %d: → %s(%s)",
                         step_counter[0],
                         serialized.get("name", "?"),
                         inp)

            def on_tool_end(self, output, **kwargs):
                out = str(output)[:120].replace("\n", " ")
                log.info("[SQL Gen]   ← 返回: %s", out)

        # 配置 callback（设置到 invoke config 中）
        callback = StepLoggerCallback()

        log.info("[SQL Gen] ─── 开始任务分解 ───")
        result = gen_with_submit.invoke(
            {"messages": gen_messages},
            config={"recursion_limit": 30, "callbacks": [callback]},
        )
        log.info("[SQL Gen] ─── 任务执行完成 ───")

        # 提取最终答案
        final_answer = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                final_answer = msg.content.strip()
                break

        sql_list = captured.get("sql_list", [])
        last_db = captured.get("last_db", "")
        generated_sql = sql_list[-1]["sql"] if sql_list else ""

        log.info("[SQL Gen] 总计 %d 步, %d 条 SQL 执行",
                 step_counter[0], len(sql_list))

        return {
            "generated_sql": generated_sql,
            "db_name": last_db or state.get("db_name", ""),
            "messages": result["messages"],
            "final_answer": final_answer,
        }

    # ---- Insight Generator Node ---------------------------------------------

    def _insight_generator_node(self, state: AgentState) -> dict:
        """数据洞察节点：自主探索数据库，生成多维度分析报告。"""
        user_query = state.get("rewritten_query", "")
        if not user_query:
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    content = m.content
                    if content.startswith("[规范化查询]"):
                        user_query = content.replace("[规范化查询] ", "")
                    else:
                        user_query = content
                    break

        log.info("[Insight] %s", user_query[:80])

        captured = {"sql_list": [], "last_db": ""}
        gen = create_react_agent(
            model=self.llm,
            tools=[
                _make_search_databases(self.store, self.embedder.model),
                _make_search_fields(self.store, self.embedder.model),
                _make_get_table_schema(),
                _make_list_databases(),
                _make_submit_sql(captured),
                _make_create_chart(),
            ],
            prompt=self.INSIGHT_PROMPT,
        )

        gen_messages = []
        history_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if len(history_msgs) > 1:
            prev = history_msgs[-2]
            gen_messages.append(HumanMessage(
                content=f"(上一轮分析: {prev.content[:150]})"
            ))
        gen_messages.append(HumanMessage(content=user_query))

        step_counter = [0]

        class InsightStepCallback(BaseCallbackHandler):
            def on_tool_start(self, serialized, input_str, **kwargs):
                step_counter[0] += 1
                inp = str(input_str)[:120].replace("\n", " ")
                log.info("[Insight]   Step %d: → %s(%s)",
                         step_counter[0],
                         serialized.get("name", "?"),
                         inp)
            def on_tool_end(self, output, **kwargs):
                out = str(output)[:100].replace("\n", " ")
                log.info("[Insight]   ← 返回: %s", out)

        log.info("[Insight] ─── 开始数据探索 ───")
        result = gen.invoke(
            {"messages": gen_messages},
            config={
                "recursion_limit": 40,
                "callbacks": [InsightStepCallback()],
            },
        )
        log.info("[Insight] ─── 探索完成 ───")

        final_answer = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                final_answer = msg.content.strip()
                break

        sql_list = captured.get("sql_list", [])
        last_db = captured.get("last_db", "")
        generated_sql = sql_list[-1]["sql"] if sql_list else ""

        log.info("[Insight] 总计 %d 步, %d 条 SQL",
                 step_counter[0], len(sql_list))

        return {
            "generated_sql": generated_sql,
            "db_name": last_db or state.get("db_name", ""),
            "messages": result["messages"],
            "final_answer": final_answer,
            "insight_report": final_answer,
        }

    # ---- Multi-Task Node ----------------------------------------------------

    def _multi_task_node(self, state: AgentState) -> dict:
        """多任务并行节点：分解 → TaskBus 事件驱动执行 → 聚合。"""
        from src.multi_task_agent import MultiTaskManager, TaskBus

        user_query = state.get("rewritten_query", "")
        if not user_query:
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    content = m.content
                    if content.startswith("[规范化查询]"):
                        user_query = content.replace("[规范化查询] ", "")
                    else:
                        user_query = content
                    break

        log.info("[Multi] 查询: %s", user_query[:80])

        mgr = MultiTaskManager(self.llm, self.store, self.embedder.model)

        # 1. 分解
        tasks = mgr.decompose(user_query)
        task_summary = ", ".join(
            f"{t['id']}[{t['type']}]" for t in tasks
        )
        log.info("[Multi] 分解为 %d 个任务: %s", len(tasks), task_summary)

        # 2. DAG 可视化（依赖关系）
        for t in tasks:
            deps = t.get("depends", [])
            dep_str = f" ← {', '.join(deps)}" if deps else ""
            log.info("[Multi]   %s: %s%s", t["id"], t.get("prompt", "")[:60], dep_str)

        # 3. TaskBus 事件驱动并行执行（自动重试 + 降级）
        bus = TaskBus(mgr)
        stats = bus.execute_all(tasks)

        # 4. 聚合
        final = mgr.aggregate(tasks)
        log.info("[Multi] 完成: %d 成功, %d 失败",
                 stats["success"], stats["failed"])

        return {
            "final_answer": final,
            "messages": [
                AIMessage(content=final),
                HumanMessage(content=user_query),
            ],
            "db_name": tasks[0].get("prompt", "") if tasks else "",
        }

    # ---- 路由函数 -----------------------------------------------------------

    @staticmethod
    def _route_after_router(state: AgentState) -> str:
        return state.get("intent", "chat")

    # ---- 辅助方法 -----------------------------------------------------------

    @staticmethod
    def _strip_tool_messages(messages):
        """过滤掉历史中的 tool 消息，保留用户消息和 AI 文本回复。"""
        cleaned = []
        for m in messages:
            if isinstance(m, ToolMessage):
                continue
            if isinstance(m, AIMessage):
                if m.content and m.content.strip():
                    cleaned.append(m)
                continue
            cleaned.append(m)
        return cleaned

    @staticmethod
    def _trim_by_tokens(messages, max_tokens=6000):
        """Token 级智能截断：从最新消息往前保留，不超过 token 上限。
        使用 tiktoken cl100k_base（GPT-4/DeepSeek 相同编码器）。
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            count = lambda text: len(enc.encode(text))
        except ImportError:
            # fallback: 中文字符 ≈ 0.5 token, 英文 ≈ 0.25 token
            count = lambda text: len(text) // 2

        # 硬上限兜底
        if len(messages) > 40:
            messages = messages[-40:]

        kept = []
        total = 0
        for msg in reversed(messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            tokens = count(content)
            if total + tokens > max_tokens:
                if not kept:
                    kept.append(msg)  # 至少保留最后一条
                break
            kept.insert(0, msg)
            total += tokens
        return kept

    # ---- 长期记忆 (向量化) -------------------------------------------------

    def _init_memory(self):
        """初始化对话记忆 Qdrant collection。"""
        try:
            self.store.create_collection("conversation_memory", vector_size=512)
        except Exception:
            pass  # 已存在

    def _store_memory(self, query: str, answer: str):
        """将本轮对话存入向量记忆库。"""
        try:
            text = f"用户: {query[:200]}\nAI: {answer[:200]}"
            vec = self.embedder.model.encode([text], normalize_embeddings=True)[0]
            import uuid
            self.store.client.upsert(
                collection_name="conversation_memory",
                points=[{
                    "id": str(uuid.uuid4()),
                    "vector": vec.tolist(),
                    "payload": {"query": query, "answer": answer, "text": text},
                }],
                wait=False,
            )
        except Exception:
            pass  # 记忆存储失败不阻塞主流程

    def _retrieve_memory(self, query: str, top_k: int = 5) -> str:
        """从向量记忆库检索与当前 query 最相关的历史对话。"""
        try:
            hits = self.store.search(
                query, self.embedder.model,
                collection_name="conversation_memory", top_k=top_k,
            )
            if not hits:
                return ""
            lines = ["[相关历史对话]"]
            for h in hits:
                lines.append(f"- {h['payload'].get('text', '')[:150]}")
            return "\n".join(lines)
        except Exception:
            return ""

    # ---- 滚动摘要 ----------------------------------------------------------

    def _update_summary(self, query: str, answer: str):
        """LLM 滚动摘要：将新对话压缩到 running_summary 中。"""
        prev = self.state.get("running_summary", "")
        if not prev:
            self.state["running_summary"] = (
                f"用户问了关于 {query[:50]} 的问题，AI 回答了相关结果。"
            )
            return
        try:
            prompt = (
                f"将以下对话要点合并为一段 ≤300 字的摘要。只保留关键数据和结论。\n"
                f"[已有摘要]: {prev}\n"
                f"[新对话] 用户: {query[:200]}\nAI: {answer[:300]}"
            )
            resp = self.llm.invoke(prompt, temperature=0.1, max_tokens=200)
            self.state["running_summary"] = resp.content.strip()[:400]
        except Exception:
            pass  # 摘要失败不阻塞

    @staticmethod
    def _extract_sql_from_text(text):
        """从 LLM 输出的文本中提取 SQL 语句。"""
        # 尝试 ```sql ... ``` 代码块
        m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # 尝试 SELECT 开头的行
        m = re.search(r"(SELECT\s+.*?)(?:;|\n\n|$)", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return ""

    # ---- 公共 API ----------------------------------------------------------

    def run(self, query, verbose=False):
        """执行用户查询。多轮对话：自动维护上下文状态。

        Args:
            query: 用户输入
            verbose: 是否打印详细日志
        """
        log.info("=" * 55)
        log.info("用户查询: %s", query)
        log.info("=" * 55)

        if verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)

        start_time = time.time()

        # 构建本轮 state
        state = {
            **self.state,
            "messages": list(self.state["messages"]) + [HumanMessage(content=query)],
            "intent": "",
            "db_name": self.state.get("db_name", ""),
            "generated_sql": "",
            "final_answer": "",
        }
        # Token 级智能截断：从最新消息往前保留，不超过 token 上限
        state["messages"] = self._trim_by_tokens(
            state["messages"], max_tokens=self.MAX_CONTEXT_TOKENS
        )

        result = self.graph.invoke(state)

        elapsed = time.time() - start_time
        log.info("-" * 55)
        log.info("总耗时 %.1fs", elapsed)

        final = result.get("final_answer", "")
        if not final:
            final = "（未能获取结果）"

        # 清理 Windows 控制台无法显示的字符
        final = final.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

        # 持久化状态
        self.state["messages"] = result["messages"]
        self.state["db_name"] = result.get("db_name", "")
        self.state["generated_sql"] = result.get("generated_sql", "")
        self.state["final_answer"] = result.get("final_answer", "")

        # 长期记忆：存储向量 + 更新摘要（异步不阻塞）
        self._store_memory(query, final)
        self._update_summary(query, final)

        return final

    def reset(self):
        """重置对话历史，开始新一轮对话。"""
        self.state = {
            "messages": [], "intent": "", "db_name": "",
            "generated_sql": "", "final_answer": "", "rewritten_query": "", "insight_report": "", "running_summary": "",
        }
        log.info("[Agent] 对话历史已重置")

    def chat_loop(self):
        """交互式多轮对话循环。输入 'exit' 或 'quit' 退出，'reset' 清空历史。"""
        print("\n" + "=" * 55)
        print("  Multi-Agent Text-to-SQL — 交互式对话")
        print("  输入 'exit' 退出 | 'reset' 清空历史")
        print("=" * 55 + "\n")

        turn = 0
        while True:
            try:
                query = input(f"[{turn+1}] 你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("再见！")
                break
            if query.lower() == "reset":
                self.reset()
                turn = 0
                print("对话历史已清空。\n")
                continue

            answer = self.run(query, verbose=False)
            print(f"\n[{turn+1}] Agent: {answer}\n")
            turn += 1


# ---- 命令行入口 ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Text-to-SQL")
    parser.add_argument("query", nargs="?", default=None, help="自然语言查询")
    parser.add_argument("-k", "--api-key", default=None, help="DeepSeek API Key")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")
    parser.add_argument("-m", "--model", default="deepseek-chat", help="LLM 模型")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互式多轮对话")

    args = parser.parse_args()
    if args.quiet:
        logging.getLogger("MultiAgent").setLevel(logging.WARNING)

    agent = SchemaAgent(api_key=args.api_key, model=args.model)

    if args.interactive:
        agent.chat_loop()
    elif args.query:
        answer = agent.run(args.query, verbose=args.verbose)
        print(f"\n{'='*55}")
        try:
            print(answer)
        except UnicodeEncodeError:
            print(answer.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
    else:
        # 无参数默认启动交互模式
        agent.chat_loop()
