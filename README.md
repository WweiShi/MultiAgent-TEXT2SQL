# Text-to-SQL 智能查询系统

基于 Multi-Agent 与 RAG 的中文自然语言转 SQL 查询系统，支持数据查询、洞察分析、图表生成与多任务并行，端到端准确率 93.5%。

## 目录

- [环境准备](#环境准备)
- [一键构建索引](#一键构建索引)
- [启动对话查询](#启动对话查询)
- [对话交互指南](#对话交互指南)
- [分步操作参考](#分步操作参考)
- [项目架构](#项目架构)
- [常见问题](#常见问题)

---

## 环境准备

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

依赖包：`langchain` `langchain-openai` `langgraph` `openai` `sentence-transformers` `qdrant-client` `matplotlib` `tiktoken`

### 2. 启动 Qdrant 向量数据库

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

> Qdrant 用于存储字段的语义向量索引，启动后监听 `localhost:6333`。如果已有 Qdrant 实例，跳过此步。

### 3. 设置 DeepSeek API Key

**Windows (PowerShell)：**
```powershell
$env:DEEPSEEK_API_KEY = "sk-your-key"
```

**Linux / macOS：**
```bash
export DEEPSEEK_API_KEY="sk-your-key"
```

> 系统使用 DeepSeek-Chat 作为 LLM，兼容 OpenAI API 协议。也可替换为其他兼容接口的模型。

### 4. 首次下载 Embedding 模型

系统使用 `BAAI/bge-small-zh-v1.5` 中文嵌入模型（512 维）。首次运行时会自动从 HuggingFace 下载（约 100MB），之后缓存到本地离线使用。

---

## 一键构建索引

完成环境准备后，使用 [index_builder.py](index_builder.py) 一条命令完成**元数据提取 → 描述生成 → 向量化 → 上传 Qdrant** 全流程。

### 全量构建（首次使用）

```bash
python index_builder.py --full --qdrant-force
```

这会扫描 `spider_data/database/` 下所有数据库，提取元数据、调用 DeepSeek 生成中文描述、向量化并存入 Qdrant。

### 仅构建指定数据库

```bash
python index_builder.py --dbs sales_db hr_1 car_1 --qdrant-force
```

### 追加新数据库（不重建已有索引）

```bash
python index_builder.py --dbs new_db --skip-extract    # 如果已手动提取过元数据
python index_builder.py --input spider_data/database/new_db/new_db.sqlite
```

### 参数说明

| 参数 | 作用 |
|------|------|
| `--full` | 扫描 `spider_data/database/` 下所有数据库 |
| `--dbs` | 指定数据库名列表（空格分隔） |
| `--input` | 单个 `.sqlite` 文件路径 |
| `--qdrant-force` | 强制重建 Qdrant collection（清空已有数据） |
| `--skip-extract` | 跳过元数据提取（已有 JSON 时使用） |
| `--skip-describe` | 跳过 LLM 描述生成（无需 API Key，但降低准确率） |
| `-k, --api-key` | DeepSeek API Key（也可设 `DEEPSEEK_API_KEY` 环境变量） |
| `-m, --model` | LLM 模型名（默认 `deepseek-chat`） |

> **注意**：如果不设 API Key 且不加 `--skip-describe`，脚本会自动跳过描述生成步骤。

---

## 启动对话查询

索引完成后，启动交互式对话：

```bash
# 交互模式
python src/schema_agent.py -i

# 单次查询
python src/schema_agent.py "统计 sales_db 中各地区的销售额"
```

### 启动参数

| 参数 | 说明 |
|------|------|
| `-i, --interactive` | 交互式多轮对话模式 |
| `-k, --api-key` | DeepSeek API Key（也可设环境变量） |
| `-m, --model` | LLM 模型名（默认 `deepseek-chat`） |
| `-v, --verbose` | 详细日志，输出每个 Agent 步骤 |
| `-q, --quiet` | 静默模式，只显示最终答案 |

直接运行 `python src/schema_agent.py`（不带参数）也会自动进入交互模式。

### 启动流程

Agent 启动时会自动：
1. 连接 Qdrant（`localhost:6333`）
2. 加载 bge-small-zh-v1.5 嵌入模型
3. 检查 `schema_fields` collection 是否存在并已索引

如果看到报错，请参考[常见问题](#常见问题)。

---

## 对话交互指南

### 基础查询

```
[1] 你: 统计每位歌手举办的演唱会数量
[1] Agent: 查询结果如下：
concert_Name        | singer_count
----------------------------------------
The Best Concert    | 3
Summer Music Fest   | 2
...（共 7 行）
```

### 支持的查询类型

系统通过 Router Node 自动识别四种意图，你无需手动切换：

| 类型 | 适用场景 | 示例 |
|------|---------|------|
| **SQL 查询** | 统计、筛选、排序等数据查询 | "查 hr_1 中薪资超过 10000 的员工" |
| **数据洞察** | 趋势分析、分布、对比 | "分析 concert_singer 的演唱会分布趋势" |
| **多任务并行** | 跨库对比、多维度同时查询 | "查 concert_singer 和 singer 分别有多少歌手并对比" |
| **聊天对话** | 问候、追问、日常交流 | "刚才查的演唱会都是哪些歌手" |

### 多轮对话

系统支持多轮上下文对话，可以基于前面的查询结果继续追问：

```
[1] 你: 统计每位歌手举办的演唱会数量
[1] Agent: ...（查询结果）

[2] 你: 这些歌手都是哪个国家的
[2] Agent: ...（自动理解"这些歌手"指代上文结果中的歌手）

[3] 你: 那有没有中国籍的？
[3] Agent: ...（继续基于上下文筛选）
```

### 对话命令

| 输入 | 作用 |
|------|------|
| `exit` 或 `quit` | 退出系统 |
| `reset` | 清空对话历史，开始新对话 |

### 图表生成

在查询中加入图表关键词，Agent 会自动生成统计图并保存为 PNG：

```
你: 查询各部门平均薪资并画柱状图
Agent: 已生成图表，保存至 output/charts/bar_20260609_143022.png
```

支持的图表类型：`饼图`/`pie`、`柱状图`/`bar`、`折线图`/`line`

---

## 分步操作参考

以下为各个步骤的独立执行方式，适合需要精细控制或排查问题的场景。

### 第一步：放置数据库文件

数据库需遵循目录约定：`spider_data/database/<数据库名>/<数据库名>.sqlite`

例如：

```
spider_data/
└── database/
    └── sales_db/
        └── sales_db.sqlite
```

### 第二步：提取元数据

```bash
python src/schema_manager.py extract spider_data/database/sales_db/sales_db.sqlite -o metadata
python src/schema_manager.py extract spider_data/database -o metadata   # 批量
```

读取内容：列名/类型/可空、主外键关系、采样值、去重数/空值数/行数、数值列的 min/max/avg。输出到 `metadata/<数据库名>.json`。

### 第三步：生成字段描述

```bash
python src/schema_manager.py describe metadata/sales_db.json
python src/schema_manager.py describe metadata/                        # 批量
```

通过 DeepSeek API 为每个字段生成中文业务描述。需要设置 `DEEPSEEK_API_KEY`。

### 第四步：构建向量索引

```bash
# 全量构建
python src/field_embedder.py metadata/ --qdrant-upload --qdrant-force

# 指定数据库
python src/field_embedder.py metadata/ --db-list sales_db hr_1 --qdrant-upload --qdrant-force

# 追加数据库（不重建已有索引）
python src/field_embedder.py metadata/new_db.json --qdrant-upload
```

### 验证索引

```bash
python src/qdrant_store.py info schema_fields
python src/schema_retriever.py "员工薪资" --db sales_db
```

---

## 项目架构

```
Rewrite → Router ──chat───→ Chat
                ├──sql────→ SQL Generator (ReAct + 6 个 Tool)
                ├──insight→ Insight Generator (6 维度自主探索)
                └──multi──→ Decomposer → TaskBus (asyncio 事件总线) → Aggregator
```

### 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 一键构建 | [index_builder.py](index_builder.py) | 一条命令完成元数据提取 → 描述 → 向量化 → 上传全流程 |
| 元数据提取 | [src/schema_manager.py](src/schema_manager.py) | 从 SQLite 提取字段级元数据，调用 DeepSeek 生成中文描述 |
| 向量化 | [src/field_embedder.py](src/field_embedder.py) | 将字段元数据转换为语义嵌入向量 |
| 向量存储 | [src/qdrant_store.py](src/qdrant_store.py) | Qdrant 向量数据库 CRUD 与检索 |
| Schema 检索 | [src/schema_retriever.py](src/schema_retriever.py) | 向量检索 + 外键扩展 + 上下文格式化 |
| 主控 Agent | [src/schema_agent.py](src/schema_agent.py) | Multi-Agent 编排，LangGraph StateGraph，6 节点 4 路由 |
| 并行引擎 | [src/multi_task_agent.py](src/multi_task_agent.py) | 任务分解 DAG + asyncio 事件总线并行调度 |
| SQL 安全 | [src/safety_checker.py](src/safety_checker.py) | 纯规则引擎，阻断写操作和多语句注入 |
| 图表生成 | [src/chart_tool.py](src/chart_tool.py) | matplotlib 图表生成（饼图/柱状图/折线图） |

### 6 个 LangChain Tool

Agent 通过 function calling 自主决定调用哪些工具：

| 工具 | 参数 | 作用 |
|------|------|------|
| `search_databases` | `query_text` | 全局检索，返回相关度最高的 Top-5 数据库 |
| `search_fields` | `query_text, db_name` | 在指定数据库中检索最相关的字段（db_name 支持逗号分隔多库） |
| `get_table_schema` | `db_name, table_name` | 读取 SQLite DDL，返回完整建表语句 |
| `list_databases` | 无 | 列出所有可用数据库 |
| `submit_sql` | `sql, db_name` | 安全检查 → 执行 SQL → 返回结果（可多次调用） |
| `create_chart` | `chart_type, data_text, title` | 生成饼图/柱状图/折线图并保存 PNG |

### 技术栈

LangGraph · LangChain · DeepSeek-Chat · BAAI/bge-small-zh-v1.5 · Qdrant · SQLite · matplotlib · asyncio

---

## 常见问题

### Q: 启动时报 "未找到 collection schema_fields"

说明还没有构建向量索引，请参考[构建向量索引](#构建向量索引)。

### Q: 启动时报 "连接 Qdrant 失败"

确认 Qdrant 容器正在运行：
```bash
docker ps | grep qdrant
```
如果没有运行，重新启动：
```bash
docker start qdrant
```

### Q: 查询结果不准确

可能原因及排查方法：
1. 没有生成字段描述（缺少业务语义）—— 运行 `python src/schema_manager.py describe metadata/<数据库名>.json`
2. 数据库名不在索引中 —— 检查是否在构建索引时漏掉了该库
3. 查看 Agent 的详细推理过程：`python src/schema_agent.py "你的查询" -v`

### Q: 如何增加新的数据库

每次新增数据库（如 `new_db.sqlite`）后：

```bash
# 1. 放好文件
mkdir -p spider_data/database/new_db
cp your_db.sqlite spider_data/database/new_db/new_db.sqlite

# 2. 一键构建索引
python index_builder.py --input spider_data/database/new_db/new_db.sqlite
```

然后重启 Agent 即可。

### Q: 如何替换 LLM 为其他模型

系统使用 OpenAI 兼容 API。修改 `src/schema_agent.py` 中的以下参数即可：

- `base_url` — API 地址
- `model` — 模型名称
- `api_key` — API Key

命令行可通过 `-m` 参数指定模型名：
```bash
python src/schema_agent.py -i -m "gpt-4o"
```
