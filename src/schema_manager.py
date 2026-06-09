"""
SchemaManager: SQLite 数据库字段级元数据提取与管理。

用法:
    manager = SchemaManager(output_dir="metadata")
    manager.fetch_metadata("path/to/db.sqlite")    # 单个文件
    manager.create_metadata("path/to/databases")   # 批量遍历文件夹
    manager.describe_single_metadata("metadata/db.json")  # LLM 增强描述
    manager.describe_all_metadata("metadata/")            # 批量 LLM 增强

依赖:
    pip install openai          # DeepSeek API 调用
    环境变量 DEEPSEEK_API_KEY   # 或通过 api_key 参数传入
"""

import os
import sqlite3
import json
import time


class SchemaManager:
    def __init__(self, output_dir="metadata", api_key=None, llm_model="deepseek-chat"):
        self.output_dir = output_dir
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.llm_model = llm_model
        self._openai_client = None
        os.makedirs(output_dir, exist_ok=True)

    # ---- LLM 增强 ---------------------------------------------------------

    def describe_single_metadata(self, json_path):
        """对单个元数据 JSON 文件调用 DeepSeek，为每个字段生成中文描述。

        必要参数：
            json_path: str  -- 由 fetch_metadata 生成的元数据 JSON 文件路径

        返回：
            原地更新 JSON 文件，每个字段新增 "description" 键。
        """
        if not self.api_key:
            raise RuntimeError(
                "未设置 DeepSeek API Key。请设置环境变量 DEEPSEEK_API_KEY "
                "或通过 SchemaManager(api_key='...') 传入。"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        db_name = metadata.get("database", os.path.basename(json_path))
        tables = metadata.get("tables", {})

        # 收集所有需要描述的字段（去重「按名称完全匹配」的跨表字段，避免重复请求）
        field_contexts = self._collect_field_contexts(tables)

        if not field_contexts:
            print(f"[{db_name}] 没有需要描述的字段")
            return json_path

        print(f"[{db_name}] 调用 LLM 为 {len(field_contexts)} 个字段生成描述...")

        # 批量调用 LLM
        descriptions = self._call_llm_for_descriptions(db_name, field_contexts)

        # 回填到 metadata 中
        self._merge_descriptions(metadata, descriptions)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        hit = sum(1 for v in descriptions.values() if v)
        print(f"[{db_name}] 已生成 {hit}/{len(descriptions)} 个字段描述")
        return json_path

    def describe_all_metadata(self, folder_path=None):
        """遍历文件夹中所有元数据 JSON，批量调用 LLM 生成字段描述。

        必要参数：
            folder_path: str  -- 元数据文件所在目录 (默认: self.output_dir)
        """
        folder = folder_path or self.output_dir
        if not os.path.isdir(folder):
            raise NotADirectoryError(f"路径不是文件夹: {folder}")

        json_files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.endswith(".json")
        ])

        if not json_files:
            print(f"在 {folder} 中未找到 .json 元数据文件")
            return []

        print(f"找到 {len(json_files)} 个元数据文件")
        results = []
        for i, path in enumerate(json_files, 1):
            try:
                self.describe_single_metadata(path)
                results.append((path, True, None))
            except Exception as e:
                results.append((path, False, str(e)))
                print(f"  [{i}/{len(json_files)}] 错误: {os.path.basename(path)} -> {e}")
            time.sleep(0.3)  # 请求间隔，避免触发限流

        ok = sum(1 for _, success, _ in results if success)
        print(f"\nLLM 描述完成: 成功 {ok}, 失败 {len(results) - ok}")
        return results

    # ---- 公共 API（原 Schema 提取）-------------------------------------------

    def fetch_metadata(self, sqlite_path):
        """提取单个 sqlite 文件的字段级元数据并保存为 JSON 文件。

        必要参数：
            sqlite_path: str  -- sqlite 数据库文件的路径

        输出：
            在 {output_dir}/{db_name}.json 中保存元数据，每个字段的注释
            以 "_comment" 键写入 JSON，标明对应哪个字段。
        """
        if not os.path.exists(sqlite_path):
            raise FileNotFoundError(f"文件不存在: {sqlite_path}")

        db_name = os.path.splitext(os.path.basename(sqlite_path))[0]
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        metadata = {
            "_comment": f"字段级元数据 -- 数据库: {db_name}",
            "database": db_name,
            "tables": {},
        }

        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()

        for (table_name,) in tables:
            table_meta = self._extract_table_meta(cursor, table_name)
            metadata["tables"][table_name] = table_meta

        conn.close()

        out_path = os.path.join(self.output_dir, f"{db_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

        print(f"[{db_name}] 元数据已保存 -> {out_path}  ({len(tables)} 张表)")
        return out_path

    def create_metadata(self, folder_path):
        """遍历文件夹，对所有 .sqlite 文件调用 fetch_metadata()。

        必要参数：
            folder_path: str  -- 包含 .sqlite 文件的文件夹路径
        """
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"路径不是文件夹: {folder_path}")

        sqlite_files = []
        for root, _dirs, files in os.walk(folder_path):
            for f in files:
                if f.endswith(".sqlite"):
                    sqlite_files.append(os.path.join(root, f))

        if not sqlite_files:
            print(f"在 {folder_path} 中未找到 .sqlite 文件")
            return []

        print(f"找到 {len(sqlite_files)} 个 .sqlite 文件，开始提取元数据...")
        results = []
        for i, path in enumerate(sorted(sqlite_files), 1):
            try:
                out = self.fetch_metadata(path)
                results.append((path, out, None))
            except Exception as e:
                results.append((path, None, str(e)))
                print(f"  [{i}/{len(sqlite_files)}] 错误: {path} -> {e}")

        ok = sum(1 for _, _, err in results if err is None)
        fail = len(results) - ok
        print(f"\n完成: 成功 {ok}, 失败 {fail}")
        return results

    # ---- LLM 私有方法 ------------------------------------------------------

    def _get_client(self):
        """延迟初始化 OpenAI 客户端。"""
        if self._openai_client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "需要安装 openai 库: pip install openai"
                )
            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
            )
        return self._openai_client

    @staticmethod
    def _collect_field_contexts(tables):
        """收集所有字段的上下文信息，用于 LLM 描述生成。

        返回: dict { "table.column": field_info }
        """
        contexts = {}
        for table_name, table_meta in tables.items():
            for col in table_meta.get("columns", []):
                key = f"{table_name}.{col['name']}"

                fk = col.get("foreign_key")
                fk_info = ""
                if fk:
                    fk_info = f"外键 -> {fk['ref_table']}.{fk['ref_column']}"

                pk_info = "主键" if col.get("is_primary_key") else ""

                samples = col.get("sample_values", [])
                sample_str = ", ".join(str(v) for v in samples[:5]) if samples else "无"

                contexts[key] = {
                    "table": table_name,
                    "column": col["name"],
                    "type": col["type"],
                    "nullable": col.get("nullable", True),
                    "primary_key": pk_info,
                    "foreign_key": fk_info,
                    "sample_values": sample_str,
                    "distinct_count": col.get("distinct_count", "?"),
                    "total_rows": col.get("total_rows", "?"),
                }
        return contexts

    def _call_llm_for_descriptions(self, db_name, field_contexts):
        """批量调用 DeepSeek 为所有字段生成中文描述。

        返回: dict { "table.column": "描述文字" }
        """
        client = self._get_client()

        # 构造紧凑的字段列表
        field_lines = []
        for key, ctx in field_contexts.items():
            tags = []
            if ctx["primary_key"]:
                tags.append(ctx["primary_key"])
            if ctx["foreign_key"]:
                tags.append(ctx["foreign_key"])
            if ctx["nullable"]:
                tags.append("可为空")

            line = (
                f"- {key} | 类型={ctx['type']} | "
                f"采样=[{ctx['sample_values']}] | "
                f"去重数={ctx['distinct_count']} | "
                f"总行数={ctx['total_rows']} | "
                f"特征={' '.join(tags) if tags else '普通字段'}"
            )
            field_lines.append(line)

        prompt = f"""你是一个数据库分析师，请为以下数据库 [{db_name}] 的字段生成简洁的中文描述。

每条描述 10-20 个字，说明该字段的业务含义。根据字段名推断含义，结合类型、采样值、主外键关系综合判断。

字段列表：
{chr(10).join(field_lines)}

请返回 JSON 格式（只返回 JSON，不要其他文字）：

{{"表名.字段名": "中文描述", ...}}"""

        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )

        raw = response.choices[0].message.content.strip()

        # 解析 LLM 返回的 JSON
        try:
            # 处理可能包裹在 ```json ... ``` 中的情况
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            descriptions = json.loads(raw)
        except json.JSONDecodeError:
            # 尝试逐行提取
            descriptions = {}
            for line in raw.split("\n"):
                line = line.strip().strip(",").strip('"').strip("'")
                if '":' in line or '": "' in line:
                    try:
                        k, v = line.split(":", 1)
                        k = k.strip().strip('"').strip("'")
                        v = v.strip().strip('"').strip("'").rstrip(",")
                        descriptions[k] = v
                    except ValueError:
                        pass

        return descriptions

    @staticmethod
    def _merge_descriptions(metadata, descriptions):
        """将 LLM 生成的描述回填到 metadata 的对应字段中。"""
        for table_name, table_meta in metadata.get("tables", {}).items():
            for col in table_meta.get("columns", []):
                key = f"{table_name}.{col['name']}"
                desc = descriptions.get(key, "")
                col["description"] = desc if desc else "（未生成描述）"

    # ---- 私有方法 ----------------------------------------------------------

    def _extract_table_meta(self, cursor, table_name):
        """提取单张表的字段级元数据。"""
        # 1. PRAGMA table_info: 字段名、类型、是否可为空、默认值、主键
        col_info = cursor.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        # (cid, name, type, notnull, dflt_value, pk)

        # 2. 外键信息
        fk_info = cursor.execute(f"PRAGMA foreign_key_list('{table_name}')").fetchall()
        fk_map = {}
        for fk in fk_info:
            from_col = fk[3]   # 本表字段名
            fk_map[from_col] = {
                "ref_table": fk[2],
                "ref_column": fk[4],
            }

        # 3. 索引信息
        idx_info = cursor.execute(f"PRAGMA index_list('{table_name}')").fetchall()
        index_map = {}
        for idx in idx_info:
            idx_name = idx[1]
            idx_cols = cursor.execute(f"PRAGMA index_info('{idx_name}')").fetchall()
            index_map[idx_name] = [col[2] for col in idx_cols]  # col[2] = 字段名

        # 4. 实际数据统计（抽样值、去重数、空值率、数值范围）
        total_rows = cursor.execute(
            f'SELECT COUNT(*) FROM "{table_name}"'
        ).fetchone()[0]

        columns = []
        for cid, name, col_type, not_null, default_val, is_pk in col_info:
            col_meta = {
                # ---- 注释 ----
                "_comment": f"表 [{table_name}] 的字段 [{name}]",
                # ---- 基础属性 ----
                "name": name,
                "type": col_type.upper() if col_type else "TEXT",
                "cid": cid,
                "nullable": not not_null,
                "default_value": default_val,
                "is_primary_key": bool(is_pk),
                "foreign_key": fk_map.get(name, None),
                # ---- 数据统计 ----
                "sample_values": self._sample_values(
                    cursor, table_name, name, n=5
                ),
                "distinct_count": 0,
                "null_count": 0,
                "total_rows": total_rows,
            }

            # 去重数与空值数
            try:
                distinct = cursor.execute(
                    f'SELECT COUNT(DISTINCT "{name}") FROM "{table_name}"'
                ).fetchone()[0]
                col_meta["distinct_count"] = distinct
            except Exception:
                col_meta["distinct_count"] = -1

            try:
                nulls = cursor.execute(
                    f'SELECT COUNT(*) FROM "{table_name}" WHERE "{name}" IS NULL'
                ).fetchone()[0]
                col_meta["null_count"] = nulls
            except Exception:
                col_meta["null_count"] = -1

            # 数值类型：提取 min / max / avg
            if self._is_numeric(col_meta["type"]):
                try:
                    stats = cursor.execute(
                        f'SELECT MIN("{name}"), MAX("{name}"), AVG("{name}") '
                        f'FROM "{table_name}"'
                    ).fetchone()
                    col_meta["min_value"] = stats[0]
                    col_meta["max_value"] = stats[1]
                    col_meta["avg_value"] = (
                        round(stats[2], 4) if stats[2] is not None else None
                    )
                except Exception:
                    col_meta["min_value"] = None
                    col_meta["max_value"] = None
                    col_meta["avg_value"] = None

            columns.append(col_meta)

        return {
            "_comment": f"表 {table_name} 共 {len(columns)} 个字段, {total_rows} 行数据",
            "row_count": total_rows,
            "columns": columns,
            "indexes": index_map,
            "column_names": [c["name"] for c in columns],
        }

    @staticmethod
    def _sample_values(cursor, table_name, col_name, n=5):
        """获取字段的前 n 条非空采样值。"""
        try:
            rows = cursor.execute(
                f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
                f'WHERE "{col_name}" IS NOT NULL LIMIT {n}'
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    @staticmethod
    def _is_numeric(col_type):
        """判断字段类型是否为数值。"""
        numeric_prefixes = (
            "INT", "INTEGER", "REAL", "FLOAT", "DOUBLE",
            "NUMERIC", "DECIMAL", "BIGINT", "SMALLINT", "TINYINT",
        )
        return col_type.upper().startswith(numeric_prefixes) if col_type else False


# ---- 命令行入口 ----

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="提取 SQLite 数据库的字段级元数据，支持 LLM 增强描述"
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # 子命令: extract (原 fetch / create 合并)
    extract = sub.add_parser("extract", help="从 sqlite 文件提取元数据")
    extract.add_argument(
        "input",
        help="sqlite 文件路径 或 包含 .sqlite 文件的文件夹路径",
    )
    extract.add_argument(
        "-o", "--output", default="metadata",
        help="元数据输出目录 (默认: metadata)",
    )

    # 子命令: describe
    describe = sub.add_parser("describe", help="调用 DeepSeek 为字段生成中文描述")
    describe.add_argument(
        "input",
        help="单个 metadata JSON 文件 或 包含 JSON 文件的文件夹路径",
    )
    describe.add_argument(
        "-k", "--api-key",
        help="DeepSeek API Key (也可通过环境变量 DEEPSEEK_API_KEY 设置)",
    )
    describe.add_argument(
        "-m", "--model", default="deepseek-chat",
        help="LLM 模型名称 (默认: deepseek-chat)",
    )

    args = parser.parse_args()

    if args.command == "describe":
        mgr = SchemaManager(api_key=args.api_key, llm_model=args.model)
        if os.path.isfile(args.input):
            mgr.describe_single_metadata(args.input)
        elif os.path.isdir(args.input):
            mgr.describe_all_metadata(args.input)
        else:
            print(f"路径不存在: {args.input}")
    else:
        # 默认: extract
        mgr = SchemaManager(output_dir=args.output)
        if os.path.isfile(args.input):
            mgr.fetch_metadata(args.input)
        elif os.path.isdir(args.input):
            mgr.create_metadata(args.input)
        else:
            print(f"路径不存在: {args.input}")
