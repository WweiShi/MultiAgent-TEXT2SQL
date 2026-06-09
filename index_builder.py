"""
一键构建索引：提取元数据 → 生成描述 → 向量化 → 上传 Qdrant。

用法:
    # 全量构建（索引所有数据库）
    python index_builder.py --full --db-dir ./my_databases

    # 构建指定数据库，跳过描述生成
    python index_builder.py --dbs sales_db hr_1 car_1 --db-dir ./my_databases --skip-describe

    # 单数据库首次导入
    python index_builder.py --input ./my_databases/sales_db/sales_db.sqlite --db-dir ./my_databases

    # 查看参数
    python index_builder.py --help
"""

import os
import sys
import argparse
import time

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.schema_manager import SchemaManager
from src.field_embedder import FieldEmbedder
from src.qdrant_store import QdrantStore

DEFAULT_DB_DIR = os.path.join(_project_root, "spider_data", "database")
METADATA_DIR = os.path.join(_project_root, "metadata")
COLLECTION_NAME = "schema_fields"


def main():
    parser = argparse.ArgumentParser(
        description="一键构建 Text-to-SQL 索引（提取 → 描述 → 向量化 → 上传 Qdrant）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python index_builder.py --full --db-dir ./my_databases                # 全量构建
  python index_builder.py --dbs sales_db hr_1 car_1 --db-dir ./my_databases  # 指定数据库
  python index_builder.py --input ./my_databases/sales_db/sales_db.sqlite --db-dir ./my_databases
  python index_builder.py --dbs sales_db --skip-describe --db-dir ./my_databases
  python index_builder.py --full --skip-extract --skip-describe --db-dir ./my_databases
        """,
    )

    # 数据源
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument(
        "--full", action="store_true",
        help="扫描 --db-dir 目录下所有数据库并构建索引",
    )
    src_group.add_argument(
        "--dbs", nargs="+", default=None,
        help="指定数据库名列表（空格分隔），如: --dbs sales_db hr_1 car_1",
    )
    src_group.add_argument(
        "--input", default=None,
        help="单个 .sqlite 文件路径，自动提取文件名作为数据库名",
    )
    parser.add_argument(
        "--db-dir", default=None,
        help="SQLite 数据库存放目录（默认: spider_data/database/）",
    )

    # 步骤控制
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="跳过元数据提取（metadata 目录已有 JSON 时使用）",
    )
    parser.add_argument(
        "--skip-describe", action="store_true",
        help="跳过 LLM 字段描述生成（无需 DeepSeek API 调用，但会降低检索准确率）",
    )

    # LLM
    parser.add_argument(
        "-k", "--api-key", default=None,
        help="DeepSeek API Key（也可通过环境变量 DEEPSEEK_API_KEY 设置）",
    )
    parser.add_argument(
        "-m", "--model", default="deepseek-chat",
        help="LLM 模型名称 (默认: deepseek-chat)",
    )

    # Qdrant
    parser.add_argument(
        "--qdrant-host", default="localhost", help="Qdrant 地址 (默认: localhost)",
    )
    parser.add_argument(
        "--qdrant-port", type=int, default=6333, help="Qdrant 端口 (默认: 6333)",
    )
    parser.add_argument(
        "--qdrant-force", action="store_true",
        help="强制重建 Qdrant collection（清空已有索引）",
    )

    # 其他
    parser.add_argument(
        "--embedding-model", default="BAAI/bge-small-zh-v1.5",
        help="Embedding 模型名 (默认: BAAI/bge-small-zh-v1.5)",
    )

    args = parser.parse_args()

    db_dir = args.db_dir or DEFAULT_DB_DIR
    db_dir = os.path.abspath(db_dir)
    if not os.path.isdir(db_dir):
        print(f"错误: 数据库目录不存在: {db_dir}")
        print("请通过 --db-dir 指定正确的 SQLite 数据库存放目录。")
        sys.exit(1)
    print(f"数据库目录: {db_dir}")

    # ---- 步骤 0: 确定数据库列表 ----

    db_names = _resolve_databases(args, db_dir)
    if not db_names:
        print("错误: 未找到需要处理的数据库。请指定 --full、--dbs 或 --input。")
        sys.exit(1)

    print(f"目标数据库 ({len(db_names)} 个): {', '.join(db_names)}")

    # ---- 检查 Qdrant ----

    store = QdrantStore(host=args.qdrant_host, port=args.qdrant_port)
    try:
        store.client.get_collections()
        print(f"Qdrant 连接成功 ({args.qdrant_host}:{args.qdrant_port})")
    except Exception as e:
        print(f"错误: 无法连接 Qdrant ({args.qdrant_host}:{args.qdrant_port})")
        print(f"  {e}")
        print("请先启动 Qdrant: docker run -d -p 6333:6333 --name qdrant qdrant/qdrant")
        sys.exit(1)

    # ---- 步骤 1: 提取元数据 ----

    if args.skip_extract:
        print("\n=== 跳过元数据提取 ===")
    else:
        print("\n=== 步骤 1/3: 提取元数据 ===")
        mgr = SchemaManager(output_dir=METADATA_DIR)
        for db_name in db_names:
            sqlite_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
            if not os.path.exists(sqlite_path):
                print(f"  [{db_name}] 文件不存在: {sqlite_path}，跳过")
                continue
            t0 = time.time()
            mgr.fetch_metadata(sqlite_path)
            elapsed = time.time() - t0
            print(f"  [{db_name}] 完成 ({elapsed:.1f}s)")

    # ---- 步骤 2: 生成描述 ----

    if args.skip_describe:
        print("\n=== 跳过字段描述生成 ===")
    else:
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("\n=== 跳过字段描述生成（未设置 DEEPSEEK_API_KEY）===")
        else:
            print("\n=== 步骤 2/3: 生成字段描述 ===")
            mgr = SchemaManager(api_key=api_key, llm_model=args.model)
            for db_name in db_names:
                json_path = os.path.join(METADATA_DIR, f"{db_name}.json")
                if not os.path.exists(json_path):
                    print(f"  [{db_name}] metadata 文件不存在，跳过")
                    continue
                t0 = time.time()
                mgr.describe_single_metadata(json_path)
                elapsed = time.time() - t0
                print(f"  [{db_name}] 完成 ({elapsed:.1f}s)")

    # ---- 步骤 3: 向量化 + 上传 Qdrant ----

    print("\n=== 步骤 3/3: 向量化 + 上传 Qdrant ===")
    embedder = FieldEmbedder(model_name=args.embedding_model)
    items = embedder.embed_selected_databases(METADATA_DIR, db_names)

    store.create_collection(
        COLLECTION_NAME,
        vector_size=embedder.vector_size,
        force=args.qdrant_force,
    )
    store.upsert_batch(items, collection_name=COLLECTION_NAME)

    # ---- 完成 ----

    info = store.collection_info(COLLECTION_NAME)
    print(f"\n{'='*55}")
    print(f"  索引构建完成！")
    print(f"  数据库数: {len(db_names)}")
    print(f"  字段总数: {info.vectors_count}")
    print(f"  启动对话: python src/schema_agent.py -i")
    print(f"{'='*55}")


def _resolve_databases(args, db_dir):
    if args.input:
        # 如果 --input 指定了完整路径，从其父目录推导 db 名
        sqlite_path = os.path.abspath(args.input)
        db_name = os.path.splitext(os.path.basename(sqlite_path))[0]
        # 让后续步骤能正确找到文件：确保 --input 的文件在 db_dir/<db_name>/ 下
        expected_dir = os.path.join(db_dir, db_name)
        expected_path = os.path.join(expected_dir, f"{db_name}.sqlite")
        if not os.path.exists(expected_path):
            os.makedirs(expected_dir, exist_ok=True)
            if os.path.abspath(sqlite_path) != os.path.abspath(expected_path):
                import shutil
                shutil.copy2(sqlite_path, expected_path)
                print(f"  已复制: {sqlite_path} → {expected_path}")
        return [db_name]

    if args.dbs:
        return list(args.dbs)

    if args.full:
        return sorted([
            d for d in os.listdir(db_dir)
            if os.path.isdir(os.path.join(db_dir, d))
            and os.path.exists(os.path.join(db_dir, d, f"{d}.sqlite"))
        ])

    return []


if __name__ == "__main__":
    main()
