# Text-to-SQL 智能查询系统

基于 Multi-Agent 与 RAG 的中文自然语言转 SQL 查询系统，支持数据查询、洞察分析、图表生成与多任务并行，端到端准确率 93.5%。

## 目录

- [环境准备](#环境准备)
- [导入新数据库](#导入新数据库)
- [生成字段描述](#生成字段描述)
- [构建向量索引](#构建向量索引)
- [启动对话查询](#启动对话查询)
- [对话交互指南](#对话交互指南)
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

## 导入新数据库

本项目基于 Spider 数据集（166 个 SQLite 数据库），你也可以导入自己的 SQLite 数据库。整个流程分为 **提取元数据 → 生成描述 → 构建索引 → 对话查询** 四步。

### 第一步：放置数据库文件

数据库需要遵循以下目录约定：

```
spider_data/database/<数据库名>/<数据库名>.sqlite
```

例如导入一个名为 `sales_db` 的数据库：

```
spider_data/
└── database/
    └── sales_db/
        └── sales_db.sqlite
```

> 系统会根据目录名识别数据库名称，`list_databases` 工具会列出所有 `spider_data/database/` 下的子目录。

### 第二步：提取元数据

```bash
# 提取单个数据库
python src/schema_manager.py extract spider_data/database/sales_db/sales_db.sqlite -o metadata

# 或批量提取整个文件夹中所有 .sqlite 文件
python src/schema_manager.py extract spider_data/database -o metadata
```

这一步会读取 SQLite 的表结构并生成元数据 JSON，包含：

| 提取内容 | 说明 |
|----------|------|
| 列名、类型、是否可空 | 来自 `PRAGMA table_info` |
| 主键、外键关系 | 来自 `PRAGMA foreign_key_list`、`PRAGMA index_list` |
| 采样值 | 每列 5 个去重非空值 |
| 统计信息 | 去重数、空值数、总行数、数值列的最大/最小/平均值 |

输出文件为 `metadata/<数据库名>.json`。

---

## 生成字段描述

元数据提取后，字段只有技术信息（名称、类型、外键等），缺少业务语义。通过 DeepSeek API 为每个字段生成中文描述：

```bash
# 为单个数据库生成描述
python src/schema_manager.py describe metadata/sales_db.json

# 或批量为整个 metadata 文件夹生成描述
python src/schema_manager.py describe metadata/

# 指定 API Key 和模型
python src/schema_manager.py describe metadata/sales_db.json -k "sk-your-key" -m "deepseek-chat"
```

描述生成后，JSON 文件中每个字段会新增 `description` 字段，例如：

```json
{
  "name": "salary",
  "type": "REAL",
  "description": "员工月薪，单位为元",
  "sample_values": [8500.0, 12000.0, 6500.0]
}
```

> **注意**：描述生成需要调用 DeepSeek API，会产生少量费用。如果跳过此步，向量检索仍可工作，但会缺失业务语义信息，可能影响查询准确率。

---

## 构建向量索引

将描述好的元数据转换为向量并存入 Qdrant，使自然语言查询能够语义匹配到正确的表和字段。

### 方式一：通过 Python 脚本（推荐）

```bash
python -c "
from src.field_embedder import FieldEmbedder
from src.qdrant_store import QdrantStore

embedder = FieldEmbedder()
store = QdrantStore()

# 将指定数据库的元数据向量化并存入 Qdrant
items = embedder.embed_selected_databases('metadata', ['sales_db', 'hr_1', 'car_1'])
store.create_collection('schema_fields', vector_size=512, force=True)
store.upsert_batch(items, collection_name='schema_fields')
print(f'已索引 {len(items)} 个字段')
"
```

### 方式二：一步步操作

```bash
# 1. 向量化单个数据库的元数据
python src/field_embedder.py metadata/sales_db.json --save sales_vectors.json

# 2. 创建 Qdrant collection（仅首次）
python src/qdrant_store.py create schema_fields --dim 512 --force

# 3. 通过代码写入 Qdrant
python -c "
from src.field_embedder import FieldEmbedder
from src.qdrant_store import QdrantStore
e = FieldEmbedder()
s = QdrantStore()
items = e.embed_metadata_json('metadata/sales_db.json')
s.upsert_batch(items)
"
```

### 索引多个数据库

```bash
python -c "
from src.field_embedder import FieldEmbedder
from src.qdrant_store import QdrantStore
e = FieldEmbedder()
s = QdrantStore()
# 列出你要索引的数据库名
db_list = ['sales_db', 'hr_1', 'concert_singer', 'car_1', 'world_1']
items = e.embed_selected_databases('metadata', db_list)
s.create_collection('schema_fields', vector_size=512, force=True)
s.upsert_batch(items)
print(f'已索引 {len(items)} 个字段，来自 {len(db_list)} 个数据库')
"
```

> 每次新增数据库后，不需要重建整个索引——可以将新库的向量追加到已有 collection 中（使用 `force=False`）。

### 验证索引

```bash
# 查看已索引的字段数量
python src/qdrant_store.py info schema_fields

# 测试检索
python src/schema_retriever.py "员工薪资" --db sales_db
```

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

# 2. 提取 + 描述
python src/schema_manager.py extract spider_data/database/new_db/new_db.sqlite -o metadata
python src/schema_manager.py describe metadata/new_db.json

# 3. 追加索引（不重建已有索引）
python -c "
from src.field_embedder import FieldEmbedder
from src.qdrant_store import QdrantStore
e = FieldEmbedder()
s = QdrantStore()
items = e.embed_metadata_json('metadata/new_db.json')
s.upsert_batch(items, collection_name='schema_fields')
print(f'已添加 {len(items)} 个字段')
"
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
