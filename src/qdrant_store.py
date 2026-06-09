"""
QdrantStore: 管理字段 embedding 在 Qdrant 中的存储与检索。

用法:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-zh")

    store = QdrantStore(host="localhost", port=6333)
    store.create_collection("schema_fields", vector_size=384)

    items = embedder.embed_metadata_json("metadata/concert_singer.json")
    store.upsert_batch(items)

    hits = store.search("每位歌手举办的演唱会数量", model, top_k=5)
    for h in hits:
        print(h["id"], h["score"])
"""

import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)


class QdrantStore:
    def __init__(self, host="localhost", port=6333, timeout=10):
        self.client = QdrantClient(host=host, port=port, timeout=timeout)

    # ---- Collection 管理 ---------------------------------------------------

    def create_collection(self, name, vector_size=384, force=False):
        """创建 Qdrant collection，使用 cosine 距离。

        参数:
            name: collection 名称
            vector_size: 向量维度
            force: 如果已存在，是否删除重建
        """
        exists = self.collection_exists(name)
        if exists:
            if force:
                self.client.delete_collection(name)
            else:
                print(f"Collection '{name}' 已存在，跳过创建")
                return False

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"Collection '{name}' 已创建 (dim={vector_size}, distance=cosine)")
        return True

    def collection_exists(self, name):
        """检查 collection 是否存在。"""
        collections = [c.name for c in self.client.get_collections().collections]
        return name in collections

    def collection_info(self, name):
        """获取 collection 信息。"""
        return self.client.get_collection(name)

    def drop_collection(self, name):
        """删除 collection。"""
        self.client.delete_collection(name)
        print(f"Collection '{name}' 已删除")

    # ---- 写入 ---------------------------------------------------------------

    def upsert_batch(self, items, collection_name="schema_fields"):
        """批量写入字段 embedding 到 Qdrant。

        参数:
            items: [(field_id, embedding_text, vector, payload), ...]
                  来自 FieldEmbedder.embed_metadata_json() 的返回结果
            collection_name: Qdrant collection 名称

        payload 存储: field_id, database, table, column, type, is_pk, is_fk,
                      ref_table, ref_column, referenced_by, description
        """
        points = []
        for field_id, text, vector, payload in items:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "field_id": field_id,
                    "embedding_text": text,
                    **payload,
                },
            )
            points.append(point)

        self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        print(f"已写入 {len(points)} 个字段向量到 '{collection_name}'")

    # ---- 检索 ---------------------------------------------------------------

    def search(self, query_text, model, collection_name="schema_fields", top_k=5,
               filter_dict=None):
        """文本查询 → 向量化 → cosine 检索 top-k。

        参数:
            query_text: 自然语言查询文本
            model: SentenceTransformer 实例
            collection_name: Qdrant collection 名称
            top_k: 返回结果数
            filter_dict: 可选，Qdrant 过滤条件，如 {"database": "concert_singer"}

        返回:
            [{"id": ..., "field_id": ..., "score": ..., "payload": {...}}, ...]
        """
        query_vector = model.encode(
            [query_text], normalize_embeddings=True
        )[0].tolist()

        # 构建过滤条件
        search_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                else:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            search_filter = Filter(must=conditions) if len(conditions) == 1 \
                else Filter(must=conditions)

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=search_filter,
        )

        return [
            {
                "id": hit.id,
                "field_id": hit.payload.get("field_id", ""),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results.points
        ]

    # ---- 精确查询 -----------------------------------------------------------

    def get_by_field_id(self, field_id, collection_name="schema_fields"):
        """按 field_id 精确查询单个字段的 payload。

        参数:
            field_id: "db_name.table_name.column_name" 格式
        """
        results = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="field_id", match=MatchValue(value=field_id))]
            ),
            limit=1,
        )[0]

        if results:
            return {"id": results[0].id, "payload": results[0].payload}
        return None

    def get_by_ref_pattern(self, table_name, collection_name="schema_fields", limit=5):
        """按表名模糊查询（用于 FK 扩展时获取被引用表的字段）。

        返回指定表中的所有字段 payload 列表。
        """
        results = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="table", match=MatchValue(value=table_name))]
            ),
            limit=limit,
        )[0]

        return [{"id": r.id, "payload": r.payload} for r in results]


# ---- 命令行入口 ----

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qdrant 存储管理")
    sub = parser.add_subparsers(dest="command")

    # 创建 collection
    create = sub.add_parser("create", help="创建 collection")
    create.add_argument("name", help="collection 名称")
    create.add_argument("--dim", type=int, default=384, help="向量维度")
    create.add_argument("--force", action="store_true", help="强制重建")

    # 查看 collection 信息
    info = sub.add_parser("info", help="查看 collection 信息")
    info.add_argument("name", help="collection 名称")

    # 删除 collection
    drop = sub.add_parser("drop", help="删除 collection")
    drop.add_argument("name", help="collection 名称")

    args = parser.parse_args()
    store = QdrantStore()

    if args.command == "create":
        store.create_collection(args.name, vector_size=args.dim, force=args.force)
    elif args.command == "info":
        info = store.collection_info(args.name)
        print(f"Collection: {args.name}")
        print(f"  vectors: {info.vectors_count}")
        print(f"  segments: {info.segments_count}")
    elif args.command == "drop":
        store.drop_collection(args.name)
    else:
        parser.print_help()
