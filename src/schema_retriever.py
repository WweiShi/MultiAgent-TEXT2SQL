"""
SchemaRetriever: 高层检索接口，封装"向量检索 + FK 扩展 + 格式化输出"。

用法:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-zh")

    retriever = SchemaRetriever(store, model)
    result = retriever.retrieve("统计每位歌手举办的演唱会数量", top_k=5)
    print(retriever.format_for_prompt(result))
"""


class SchemaRetriever:
    def __init__(self, qdrant_store, embedder_model):
        self.store = qdrant_store
        self.model = embedder_model

    # ---- 核心检索流程 -------------------------------------------------------

    def retrieve(self, query, collection_name="schema_fields", top_k=5, db_name=None):
        """完整检索流程：向量检索 → FK 扩展 → 去重排序。

        参数:
            query: 自然语言问题
            collection_name: Qdrant collection 名称
            top_k: 向量检索 top-k
            db_name: 可选，限制只在指定数据库中检索

        返回:
            {
                "query": 原始查询,
                "top_k_hits": [原始 top-k 命中],
                "expanded_hits": [FK 扩展补充的命中],
                "all_fields": [去重后的完整字段列表],
            }
        """
        # 第一步：向量检索 (可指定数据库过滤)
        filter_dict = {"database": db_name} if db_name else None
        raw_hits = self.store.search(
            query, self.model, collection_name, top_k=top_k, filter_dict=filter_dict
        )

        # 第二步：FK 扩展
        expanded = self._expand_foreign_keys(raw_hits, collection_name)

        # 第三步：去重合并
        seen = set()
        all_fields = []

        for h in raw_hits:
            fid = h["field_id"]
            if fid not in seen:
                seen.add(fid)
                all_fields.append(h)

        for h in expanded:
            fid = h["field_id"]
            if fid not in seen:
                seen.add(fid)
                all_fields.append(h)

        return {
            "query": query,
            "top_k_hits": raw_hits,
            "expanded_hits": expanded,
            "all_fields": all_fields,
            "field_count": len(all_fields),
        }

    # ---- FK 扩展逻辑 -------------------------------------------------------

    def _expand_foreign_keys(self, hits, collection_name):
        """对每个命中字段做 FK 展开: 前向 + 反向。

        前向展开: 命中字段有 FK → 补入被引用表的 PK + 第一个语义字段
        反向展开: 命中字段是 PK → 补入引用它的 FK 字段
        """
        expanded = []

        for hit in hits:
            payload = hit.get("payload", {})

            # 前向展开: 此字段是 FK，补入被引用表的关键字段
            if payload.get("is_fk"):
                ref_table = payload.get("ref_table")
                ref_column = payload.get("ref_column")
                if ref_table and ref_column:
                    forward = self._lookup_referenced_fields(
                        ref_table, ref_column, collection_name
                    )
                    expanded.extend(forward)

            # 反向展开: 此字段是 PK，补入引用它的 FK 字段
            if payload.get("is_pk"):
                referenced_by = payload.get("referenced_by", [])
                for caller in referenced_by[:2]:  # 最多取 2 个引用者
                    # caller 格式: "table_name.column_name"
                    caller_table = caller.split(".")[0]
                    caller_col = caller.split(".")[1]
                    # 构建 field_id: "db.table.column"
                    db = payload.get("database", "")
                    field_id = f"{db}.{caller_table}.{caller_col}"
                    backward = self.store.get_by_field_id(field_id, collection_name)
                    if backward:
                        expanded.append({
                            "source": "reverse_fk",
                            "field_id": field_id,
                            "payload": backward["payload"],
                        })

        return expanded

    def _lookup_referenced_fields(self, ref_table, ref_column, collection_name):
        """获取被引用表的关键字段: PK 列 + 语义列。

        返回被引用表中作为 JOIN 目标的关键字段列表。
        """
        results = []

        # 1. 精确获取被引用的 PK 字段
        ref_fields = self.store.get_by_ref_pattern(
            ref_table, collection_name, limit=20
        )

        for rf in ref_fields:
            p = rf["payload"]
            if p.get("column") == ref_column or p.get("is_pk"):
                results.append({
                    "source": "forward_fk",
                    "field_id": p.get("field_id", ""),
                    "payload": p,
                })

        # 2. 额外补一个语义字段（非 PK、非 FK 的第一个描述字段，如 name/title）
        for rf in ref_fields:
            p = rf["payload"]
            if p.get("is_pk") or p.get("is_fk"):
                continue
            col = p.get("column", "").lower()
            # 优先匹配语义明显的列名
            if any(kw in col for kw in ("name", "title", "label", "description")):
                results.append({
                    "source": "forward_fk_semantic",
                    "field_id": p.get("field_id", ""),
                    "payload": p,
                })
                break

        return results

    # ---- 格式化输出 ---------------------------------------------------------

    @staticmethod
    def format_for_prompt(result):
        """将检索结果格式化为可直接喂给 LLM 的文本。

        输出格式:
            ### 相关字段 (top-5 检索 + FK 扩展)
            - concert.concert_Name: 演唱会名称。类型 TEXT。
            - singer.Name: 歌手姓名。类型 TEXT，主键。
            ...
        """
        lines = [
            f"### 相关字段 (检索到 {result['field_count']} 个字段)",
            f"查询: {result['query']}",
            "",
        ]

        # 按表名分组
        by_table = {}
        for f in result["all_fields"]:
            p = f["payload"]
            t = p.get("table", "?")
            if t not in by_table:
                by_table[t] = []
            by_table[t].append(f)

        for table, fields in sorted(by_table.items()):
            lines.append(f"**{table}**")
            for f in fields:
                p = f["payload"]
                desc = p.get("description", "")
                col_type = p.get("type", "")
                tags = []
                if p.get("is_pk"):
                    tags.append("PK")
                if p.get("is_fk"):
                    tags.append(f"FK→{p.get('ref_table', '?')}.{p.get('ref_column', '?')}")
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                lines.append(f"  - {p['column']} ({col_type}){tag_str}: {desc}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_compact(result):
        """紧凑格式输出，适合终端查看。"""
        lines = [f"查询: {result['query']}"]
        lines.append(f"命中 {result['field_count']} 个字段:")
        for f in result["all_fields"]:
            p = f["payload"]
            source = f.get("source", "vector")
            fid = f.get("field_id", p.get("field_id", ""))
            score = f.get("score")
            score_str = f" [{score:.4f}]" if score is not None else ""
            lines.append(f"  ({source}) {fid}{score_str}  -- {p.get('description', '')}")
        return "\n".join(lines)


# ---- 命令行入口 ----

if __name__ == "__main__":
    import argparse
    import json
    from qdrant_store import QdrantStore
    from field_embedder import FieldEmbedder

    parser = argparse.ArgumentParser(
        description="Schema 字段检索器 (向量检索 + FK 扩展)"
    )
    parser.add_argument("query", help="自然语言查询问题")
    parser.add_argument(
        "--collection", default="schema_fields",
        help="Qdrant collection 名称",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="向量检索 top-k",
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Qdrant 地址",
    )
    parser.add_argument(
        "--port", type=int, default=6333,
        help="Qdrant 端口",
    )
    parser.add_argument(
        "--model", default="BAAI/bge-small-zh-v1.5",
        help="embedding 模型",
    )
    parser.add_argument(
        "--db", default=None,
        help="限制在指定数据库中检索",
    )

    args = parser.parse_args()

    embedder = FieldEmbedder(model_name=args.model)
    store = QdrantStore(host=args.host, port=args.port)

    if not store.collection_exists(args.collection):
        print(f"错误: Collection '{args.collection}' 不存在，请先创建并导入数据")
        exit(1)

    retriever = SchemaRetriever(store, embedder.model)
    result = retriever.retrieve(args.query, args.collection, top_k=args.top_k, db_name=args.db)

    print(retriever.format_compact(result))
    print()
    print(retriever.format_for_prompt(result))
