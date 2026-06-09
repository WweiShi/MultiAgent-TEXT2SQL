"""
FieldEmbedder: 将 metadata JSON 中的每个字段转换为自然语言语义片段，并向量化。

用法:
    embedder = FieldEmbedder(model_name="BAAI/bge-small-zh")
    items = embedder.embed_metadata_json("metadata/concert_singer.json")
    # items = [(field_id, text, vector, payload), ...]
"""

import json
import os
from sentence_transformers import SentenceTransformer


class FieldEmbedder:
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5", offline=True):
        # 离线模式下直接使用本地缓存，避免因网络不通而卡死
        load_kwargs = {}
        if offline:
            load_kwargs = {"local_files_only": True}
        self.model = SentenceTransformer(model_name, **load_kwargs)
        self.vector_size = self.model.get_sentence_embedding_dimension()

    # ---- 公共 API ----------------------------------------------------------

    def embed_metadata_json(self, json_path):
        """读取单个 metadata JSON 文件，为所有字段生成 embedding。

        返回: [(field_id, embedding_text, vector, payload), ...]
        """
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        db_name = metadata.get("database", "")
        tables = metadata.get("tables", {})

        # 构建反向 FK 索引: { "table_name.column_name": ["引用者1", ...] }
        reverse_fk = self._build_reverse_fk_index(tables)

        all_texts = []
        all_payloads = []

        for table_name, table_meta in tables.items():
            for col_meta in table_meta.get("columns", []):
                field_id = f"{db_name}.{table_name}.{col_meta['name']}"
                text = self.build_embedding_text(table_name, col_meta, reverse_fk)
                payload = self._build_payload(db_name, table_name, col_meta, reverse_fk)

                all_texts.append(text)
                all_payloads.append(payload)

        # 批量向量化
        vectors = self.model.encode(all_texts, normalize_embeddings=True)

        return [
            (field_id, text, vector.tolist(), payload)
            for field_id, text, vector, payload
            in zip(
                [f"{db_name}.{t}.{c['name']}" for t, m in tables.items() for c in m["columns"]],
                all_texts,
                vectors,
                all_payloads,
            )
        ]

    def embed_selected_databases(self, metadata_folder, db_list):
        """仅处理指定数据库的 metadata JSON 文件。

        参数:
            metadata_folder: metadata JSON 文件所在文件夹路径
            db_list: 数据库名列表，如 ["concert_singer", "pets_1"]
        """
        all_items = []
        for db_name in db_list:
            json_path = os.path.join(metadata_folder, f"{db_name}.json")
            if not os.path.exists(json_path):
                print(f"  [{db_name}] metadata 文件不存在: {json_path}")
                continue
            items = self.embed_metadata_json(json_path)
            all_items.extend(items)
            print(f"  [{db_name}] {len(items)} 个字段已向量化")
        print(f"总计: {len(all_items)} 个字段")
        return all_items

    # ---- 核心：生成 embedding 文本 ------------------------------------------

    @staticmethod
    def build_embedding_text(table_name, col_meta, reverse_fk):
        """为单个字段构造富语义的自然语言 embedding 文本。

        格式:
            [表名]表的[字段名]字段，类型[TYPE]。[角色标签]。[FK方向]。[被引用方向]。
            业务含义: [LLM描述]。关键采样: [v1, v2]。
        """
        col_name = col_meta["name"]
        col_type = col_meta["type"]
        desc = col_meta.get("description", "")

        # ---- 角色标签 ----
        roles = []
        if col_meta.get("is_primary_key"):
            roles.append("主键")
        fk = col_meta.get("foreign_key")
        if fk:
            roles.append("外键")
        if not roles:
            roles.append("普通字段")

        # ---- FK 方向文本 ----
        fk_text = ""
        if fk:
            fk_text = f"外键关联到{fk['ref_table']}表的{fk['ref_column']}字段。"

        # ---- 反向引用文本 ----
        key = f"{table_name}.{col_name}"
        refs = reverse_fk.get(key, [])
        ref_text = ""
        if refs:
            ref_list = "、".join(refs)
            ref_text = f"被以下字段引用: {ref_list}。"

        # ---- 采样值 ----
        samples = col_meta.get("sample_values", [])
        sample_str = "、".join(str(v) for v in samples[:5]) if samples else "无"

        # ---- 组装 ----
        parts = [
            f"{table_name}表的{col_name}字段，类型为{col_type}。{'，'.join(roles)}。",
            fk_text,
            ref_text,
            f"业务含义: {desc}。",
            f"关键采样: {sample_str}。",
        ]
        return "".join(parts)

    # ---- Payload 构造 -----------------------------------------------------

    @staticmethod
    def _build_payload(db_name, table_name, col_meta, reverse_fk):
        """构造 Qdrant payload，存储结构化字段信息供 FK 扩展使用。"""
        fk = col_meta.get("foreign_key")
        key = f"{table_name}.{col_meta['name']}"

        return {
            "database": db_name,
            "table": table_name,
            "column": col_meta["name"],
            "type": col_meta["type"],
            "is_pk": col_meta.get("is_primary_key", False),
            "is_fk": fk is not None,
            "ref_table": fk["ref_table"] if fk else None,
            "ref_column": fk["ref_column"] if fk else None,
            "referenced_by": reverse_fk.get(key, []),
            "description": col_meta.get("description", ""),
        }

    # ---- 辅助方法 ----------------------------------------------------------

    @staticmethod
    def _build_reverse_fk_index(tables):
        """构建反向 FK 索引: { "ref_table.ref_column": ["引用者Table.Column", ...] }"""
        reverse_fk = {}
        for table_name, table_meta in tables.items():
            for col_meta in table_meta.get("columns", []):
                fk = col_meta.get("foreign_key")
                if fk:
                    ref_key = f"{fk['ref_table']}.{fk['ref_column']}"
                    caller = f"{table_name}.{col_meta['name']}"
                    if ref_key not in reverse_fk:
                        reverse_fk[ref_key] = []
                    reverse_fk[ref_key].append(caller)
        return reverse_fk


# ---- 命令行入口 ----

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="将 metadata JSON 中的字段转换为向量 embedding，并可上传至 Qdrant"
    )
    parser.add_argument(
        "input",
        help="单个 metadata JSON 文件路径 或 metadata 文件夹路径",
    )
    parser.add_argument(
        "--db-list", nargs="+", default=None,
        help="指定数据库名列表（仅处理这些数据库），如: --db-list concert_singer pets_1",
    )
    parser.add_argument(
        "--model", default="BAAI/bge-small-zh-v1.5",
        help="sentence-transformers 模型名 (默认: BAAI/bge-small-zh-v1.5)",
    )
    parser.add_argument(
        "--save", default=None,
        help="将向量序列化保存到指定 JSON 文件中",
    )
    parser.add_argument(
        "--qdrant-upload", action="store_true",
        help="向量化后直接上传到 Qdrant",
    )
    parser.add_argument(
        "--qdrant-collection", default="schema_fields",
        help="Qdrant collection 名称 (默认: schema_fields)",
    )
    parser.add_argument(
        "--qdrant-force", action="store_true",
        help="强制重建 Qdrant collection（已有数据将被清除）",
    )
    parser.add_argument(
        "--qdrant-host", default="localhost",
        help="Qdrant 地址 (默认: localhost)",
    )
    parser.add_argument(
        "--qdrant-port", type=int, default=6333,
        help="Qdrant 端口 (默认: 6333)",
    )

    args = parser.parse_args()
    embedder = FieldEmbedder(model_name=args.model)
    print(f"模型: {args.model}, 向量维度: {embedder.vector_size}")

    if os.path.isfile(args.input):
        items = embedder.embed_metadata_json(args.input)
        print(f"已处理 {len(items)} 个字段")
        for fid, text, vec, payload in items[:3]:
            print(f"  [{fid}] {text[:120]}...")

    elif os.path.isdir(args.input):
        if args.db_list:
            items = embedder.embed_selected_databases(args.input, args.db_list)
        else:
            # 处理文件夹中所有 JSON
            all_items = []
            json_files = sorted([
                os.path.join(args.input, f) for f in os.listdir(args.input)
                if f.endswith(".json")
            ])
            for jf in json_files:
                db_items = embedder.embed_metadata_json(jf)
                all_items.extend(db_items)
                print(f"  [{os.path.basename(jf)}] {len(db_items)} 个字段")
            items = all_items
            print(f"总计: {len(items)} 个字段")

    if args.save:
        serializable = [
            {"id": fid, "text": text, "vector": vec, "payload": payload}
            for fid, text, vec, payload in items
        ]
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"向量已保存到: {args.save}")

    if args.qdrant_upload:
        from src.qdrant_store import QdrantStore
        store = QdrantStore(host=args.qdrant_host, port=args.qdrant_port)
        store.create_collection(
            args.qdrant_collection,
            vector_size=embedder.vector_size,
            force=args.qdrant_force,
        )
        store.upsert_batch(items, collection_name=args.qdrant_collection)
