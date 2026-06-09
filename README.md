# Text-to-SQL 智能查询系统

基于 Multi-Agent 与 RAG 的中文自然语言转 SQL 查询系统，支持数据查询、洞察分析、图表生成与多任务并行，端到端准确率 93.5%。

## 快速开始

```powershell
# 1. 安装依赖
pip install langchain langchain-openai langgraph openai sentence-transformers qdrant-client matplotlib tiktoken

# 2. 启动 Qdrant
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# 3. 设置 API Key
$env:DEEPSEEK_API_KEY = "sk-your-key"

# 4. 构建索引（首次）
python src/schema_manager.py extract spider_data/database -o metadata
python src/schema_manager.py describe metadata/
python -c "from src.field_embedder import FieldEmbedder; from src.qdrant_store import QdrantStore; e=FieldEmbedder(); s=QdrantStore(); items=e.embed_selected_databases('metadata', ['concert_singer','hr_1','car_1','world_1']); s.create_collection('schema_fields',512,force=True); s.upsert_batch(items)"

# 5. 启动
python src/schema_agent.py -i
```

## 功能

| 功能 | 示例 |
|------|------|
| 自然语言查询 | "统计每位歌手举办的演唱会数量" |
| 数据洞察 | "分析 hr_1 的员工薪资分布趋势" |
| 图表生成 | "查询各部门平均薪资并画柱状图" |
| 跨库对比 | "查 concert_singer 和 singer 分别有多少歌手并对比" |
| 多轮对话 | 先问"有哪些歌手"→追问"这些歌手都是哪个国家的" |

## 架构

```
Rewrite → Router ──chat───→ Chat
                ├──sql────→ SQL Generator (ReAct + 6 Tool)
                ├──insight→ Insight Generator (ReAct + 6 Tool)
                └──multi──→ Decomposer → TaskBus (asyncio 事件总线) → Aggregator
```

| 路由 | 能力 |
|------|------|
| Chat | 多轮对话，上下文感知 |
| SQL Generator | ReAct 自主检索 Schema + 生成执行 SQL |
| Insight Generator | 6 维度自主探索，生成洞察报告 |
| Multi-Task | LLM 拆解 DAG → asyncio 并行调度 → 聚合 |

## 技术栈

LangGraph · LangChain · DeepSeek-Chat · bge-small-zh-v1.5 · Qdrant · SQLite · matplotlib · asyncio

## 项目结构

```
src/
├── schema_agent.py       # Multi-Agent 主控
├── multi_task_agent.py    # TaskBus 并行引擎
├── schema_manager.py     # 元数据提取 + LLM 描述
├── field_embedder.py     # Embedding 向量化
├── qdrant_store.py       # Qdrant 存储检索
├── safety_checker.py     # SQL 安全检查
└── chart_tool.py          # 图表生成
test_runner.py            # 测试框架
```
