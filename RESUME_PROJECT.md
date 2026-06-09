# 基于 Multi-Agent 与 RAG 的智能 Text-to-SQL 系统

## 项目概述

设计并实现了一个端到端的自然语言转 SQL 查询系统，结合 **RAG 向量检索**与 **LangGraph 多智能体协作架构**，用户以中文自然语言提问即可自动完成"查询改写 → 意图路由 → 数据库定位 → Schema 检索 → SQL 生成 → 安全检查 → 执行纠错"的完整链路。支持**单轮多库对比查询**、**多轮上下文对话**、**数据洞察分析**、**统计图表生成**和**多任务并行分发**。在 Spider 数据集的 **15 个数据库** 上达到 **93.5% 端到端准确率**。

## 核心亮点

### 1. 字段级 RAG 检索与 Schema 理解

- 对 **166 个 SQLite 数据库、1000+ 张表** 提取字段级元数据（类型、主外键、采样值、去重数、空值率、数值范围），并通过 **DeepSeek API** 为每个字段自动生成中文语义描述
- 设计**双向 FK 感知的 Embedding 策略**：向量化文本同时包含 FK 引用和被引用方向，cosine 检索时自动命中 JOIN 链路两端字段
- **两阶段检索消除跨库污染**：`search_databases` 全局检索按数据库聚合相关度→`search_fields` 限库精确检索（`db_name` 支持逗号分隔多库，底层 `MatchAny` 过滤），跨库误检索率 **5%→0%**
- 使用 **BAAI/bge-small-zh-v1.5**（512维）本地离线嵌入 + **Qdrant** 向量存储，300+ 字段毫秒级检索

### 2. 四路 Multi-Agent 协作架构

基于 **LangGraph StateGraph** 构建 6 节点 4 路由多智能体流水线：

```
Rewrite → Router ──chat───→ Chat
                ├──sql────→ SQL Generator (ReAct + 6 Tool)   — 精准查询
                ├──insight→ Insight Generator (ReAct + 6 Tool) — 自主洞察
                └──multi──→ Decomposer → TaskBus → Aggregator — 多任务并行
```

| 路由 | 能力 | 关键指标 |
|------|------|---------|
| SQL Generator | ReAct 自主调用工具，生成+执行 SQL | 单轮最多 24 条 SQL |
| Insight Generator | 6 维度自主探索（概览/分布/分类/时间/TopN/占比） | 单轮 7-18 条 SQL |
| Multi-Task | LLM 分解 DAG → TaskBus 事件驱动并行 → 聚合 | 子任务成功率 100% |
| Chat | 多轮对话，上下文感知 | — |

**TaskBus 异步事件总线调度机制**：

LLM 分析查询自动拆解为带依赖关系的子任务 DAG（如：A 查库1、B 查库2、C 对比 A&B）。TaskBus 基于 `asyncio` 实现事件驱动并行调度：

- **依赖图预计算**：`_build_dependency_graph()` 一次扫描构建 `downstream_map`（被依赖者→下游任务列表）和 `pending_deps`（每任务未完成依赖计数），O(n)
- **事件总线**：`asyncio.Queue` 作为任务完成事件通道，任务完成→`bus.put()`→消费消息时只通知直接下游，O(1) 而非 O(n²) 扫描
- **LLM 调用非阻塞**：同步 LLM 调用通过 `asyncio.to_thread()` 放入线程池，不阻塞协程事件循环
- **超时保护**：`asyncio.wait_for(task, timeout=120s)` 单任务超时自动降级，`wait_for(bus.get(), timeout=300s)` 总线空等兜底
- **容错**：失败自动重试 2 次后降级标记，下游依赖失败也正常触发（收到"状态: 失败"），单节点崩溃不阻塞整体流程

### 3. 数据洞察、安全校验与可视化

**Data Insight**：Agent 自主探索数据库，从多维度生成分析报告——数据概览、分布分析（66% 员工薪资<8,000）、分类对比（Executive 薪资是 Shipping 的 5.6 倍）、区间占比等。

**SQL 安全引擎**：自研纯规则 SafetyChecker，零 LLM 调用。拒绝 INSERT/UPDATE/DELETE/DROP、检测 `;` 多语句注入、拦截 ATTACH/PRAGMA 等危险函数。阻断率 **100%**，实测成功将 `DELETE FROM singer` 自动修正为 `SELECT * FROM singer`。

**可视化制图**：基于 matplotlib，Agent 根据数据自主选择饼图/柱状图/折线图，中文渲染。

### 4. 查询改写与多轮上下文对话

**Rewite Node** 作为所有请求统一入口，基于对话历史进行智能规范化：

| 改写维度 | 原始输入 | 改写输出 |
|---------|---------|---------|
| 去噪 | "帮我查一下那个，就是所有员工的薪资是多少啊" | "查询所有员工的薪资" |
| 指代消解 | "那这些歌手都是哪个国家的" | "查询 Justin Brown 等歌手的国籍" |
| 标准化 | "马力超过一百五的车型" | "马力超过 150 的车型" |

**多层上下文管理**：三层记忆架构解决长对话中信息衰减问题：

| 层级 | 机制 | 作用 |
|------|------|------|
| Token 截断 | tiktoken `cl100k_base` 编码器精确计数，从最新消息往前保留至 6000 token，替代硬截断 20 轮 | 窗口压缩 |
| 向量记忆 | 每轮对话存入 Qdrant `conversation_memory` collection，查询时语义检索 Top-5 相关历史轮次注入 Prompt | 长期检索 |
| 滚动摘要 | LLM 将历史对话合并压缩为 ≤300 字摘要（`running_summary`），持续更新，始终注入上下文 | 信息压缩 |

Rewrite Node 入口处自动拼接三层上下文：`最近消息 + 向量检索的相关历史 + 滚动摘要`，Tool Message 自动过滤避免多轮累积的 API 400 错误。

## 技术栈

| 层 | 技术选型 |
|----|---------|
| Agent 框架 | LangGraph (StateGraph), LangChain |
| LLM | DeepSeek-Chat (OpenAI-compatible API) |
| 向量化 | BAAI/bge-small-zh-v1.5 (512d, 本地离线) |
| 向量数据库 | Qdrant (cosine 相似度 + MatchAny/MatchValue payload 过滤) |
| 数据库 | SQLite (166 个, Spider 数据集) |
| 元数据提取 | PRAGMA table_info / foreign_key_list + SQL 聚合统计 |
| 安全引擎 | 自研 SafetyChecker（纯规则引擎，零 LLM 调用） |

## 关键指标

| 指标 | 数值 |
|------|------|
| 测试数据库数 | 15 个（覆盖音乐、交通、HR、教育、地理、汽车、航班等领域） |
| 测试题目总数 | **62 题**（基础 40 + 跨库 10 + 复杂条件 12） |
| Router 意图分类准确率 | **100%** (62/62) |
| SQL 语法有效率 | **100%** (62/62) |
| 安全检查通过率 | **100%** |
| 端到端完整准确率 | **93.5%** (58/62) |
| 复杂条件查询准确率 | **100%** (12/12) |
| 多库对比查询准确率 | **100%** (10/10) |
| 单轮最大 SQL 执行数 | 24 条（Agent 自主探索） |
| 危险 SQL 阻断率 | **100%** |
| 平均查询耗时 | ~14s |

> 未满分题目均为 Spider 数据集自身的数据完整性问题（flight_2 机场代码含前导空格、bike_1 日期范围不重叠），非系统 SQL 生成逻辑缺陷。复杂查询中 Agent 能自主发现数据不满足条件（如无冬季数据、无2000年后员工），正确解释而非强行返回错误结果。

## 项目结构

```
src/
├── schema_manager.py     # SQLite → JSON 元数据提取 + LLM 描述生成
├── field_embedder.py     # 字段 Embedding 文本构造 + 向量化
├── qdrant_store.py       # Qdrant 存储层 (CRUD + 过滤检索)
├── schema_retriever.py   # 检索器 (FK 扩展 + 上下文格式化)
├── schema_agent.py       # Multi-Agent LangGraph (Router + Chat + SQL + Insight + Multi)
├── multi_task_agent.py    # 多任务分解与 Fan-out 并行执行
├── safety_checker.py     # SQL 安全检查规则引擎
└── chart_tool.py          # matplotlib 图表生成 (饼图/柱状图/折线图)
test_runner.py            # 自动化测试框架
docs/                     # 设计文档
output/charts/            # 生成的图表文件
test_questions_20.md      # 第一批测试题 (20 题)
test_questions_set2.md    # 第二批测试题 (20 题)
test_questions_crossdb.md # 跨库调用测试 (20 题)
test_questions_complex.md # 复杂条件查询测试 (50 题)
```

## 工具目录

系统包含 6 个 LangChain Tool，由 ReAct Agent 的 LLM 通过 function calling 自主决定调用时机、参数和次数，Python 代码不控制调用逻辑。

| # | 工具 | 参数 | 作用 |
|---|------|------|------|
| 1 | `search_databases` | `query_text` | 全局向量检索 → 按数据库聚合相关度分数 → 返回 Top-5 数据库排名 |
| 2 | `search_fields` | `query_text, db_name` | Qdrant 字段检索，`db_name` 支持逗号分隔多库（`MatchAny` 过滤），返回字段名/类型/描述/主外键 |
| 3 | `get_table_schema` | `db_name, table_name` | 从 SQLite `sqlite_master` 读取 DDL，返回完整建表语句 |
| 4 | `list_databases` | 无 | 列出所有可用数据库 |
| 5 | `submit_sql` | `sql, db_name` | **内联三合一**：SafetyChecker 校验 → sqlite3 执行 → 返回结果，可多次调用 |
| 6 | `create_chart` | `chart_type, data_text, title` | 调用 matplotlib 生成饼图/柱状图/折线图，保存为 PNG |

**典型调用链**：

```
LLM → search_databases("员工薪资")   → hr_1 (2.72)
LLM → search_fields("员工薪资", db_name="hr_1")  → employees.FIRST_NAME, ...
LLM → get_table_schema("hr_1", "employees")  → CREATE TABLE ...
LLM → submit_sql("SELECT FIRST_NAME, SALARY FROM employees", "hr_1")
       └─ 内部: SafetyChecker.check() → sqlite3.execute() → 107 行
LLM → 直接输出: "共 107 名员工，平均薪资 6,462..."
```

**各路由可用工具**：

| 路由 | 工具集 |
|------|--------|
| SQL Generator | 全部 6 个 |
| Insight Generator | 全部 6 个（Prompt 引导多维度探索） |
| Multi-Task | 子任务按类型分配：sql_query→前5个, chart→create_chart, synthesize→纯LLM |

## 技术栈明细

| 分类 | 技术 | 用途 |
|------|------|------|
| **Agent 编排** | LangGraph (StateGraph + Send) | 6 节点 4 路由：Router → Chat / SQL / Insight / Multi |
| | LangChain (create_react_agent) | SQL / Insight / Multi 子 Agent 的 ReAct 推理循环 |
| | asyncio (TaskBus 事件总线) | 异步事件驱动 + 依赖图 O(1) 通知 + 超时保护 + 自动重试降级 |
| **LLM** | DeepSeek-Chat | 改写、路由、SQL生成、任务分解、洞察分析、结果综合 |
| | OpenAI-compatible API | 统一 LLM 调用接口 |
| **RAG / 向量检索** | BAAI/bge-small-zh-v1.5 | 本地离线中文嵌入模型（512 维） |
| | Sentence-Transformers | 模型加载与推理框架 |
| | Qdrant | 向量数据库（cosine + MatchAny/MatchValue payload 过滤） |
| **数据层** | SQLite3 (166 个库) | Spider 数据集，只读模式连接 |
| | PRAGMA table_info / foreign_key_list | Schema 提取 |
| | SQL 聚合函数 + 窗口函数 | 字段级数据统计与复杂查询 |
| **安全** | 自研 SafetyChecker（规则引擎） | SQL 注入检测、写操作阻断、危险函数拦截 |
| **可视化** | matplotlib (pie/bar/line) | 统计图表生成，中文渲染，Agent 自主选择图表类型 |
| **元数据** | DeepSeek API | 为每个字段自动生成中文语义描述 |
| **测试** | 自研 test_runner + 分层测试集 | 4 类测试共 62 题，自动化准确率统计 |

## 一句话亮点总结

1. **RAG 检索**：字段级细粒度元数据 + 双向 FK 感知 Embedding + 两阶段数据库定位，跨库误检索率 0%
2. **Multi-Agent 架构**：6 节点 4 路由 StateGraph + ReAct 工具调用 + TaskBus asyncio 事件总线 DAG 并行调度
3. **数据洞察 + 安全 + 可视化**：6 维度自主探索 + 纯规则安全引擎 100% 阻断 + matplotlib 图表生成
4. **上下文管理**：Token 级截断 + 向量语义记忆 + LLM 滚动摘要三层架构，长对话信息零衰减 + 指代消解 + 标准化改写
