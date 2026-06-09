"""
SQL 安全检查器：规则引擎，不依赖 LLM。

检查项:
- 只读限制（仅允许 SELECT）
- 多语句注入检测
- 危险函数/关键字检测
- 代码规范检查
"""

import re


class SafetyChecker:
    # 被禁止的关键字（不区分大小写）
    FORBIDDEN_STATEMENTS = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
        "TRUNCATE", "CREATE", "REPLACE",
    ]

    # 危险的 SQLite 特定命令
    DANGEROUS_COMMANDS = [
        "ATTACH", "DETACH", "PRAGMA", "LOAD_EXTENSION",
        "SAVEPOINT", "RELEASE", "ROLLBACK", "BEGIN", "COMMIT",
        "VACUUM", "REINDEX",
    ]

    # 编码规范问题（warning 级别，不阻断）
    CODE_QUALITY_CHECKS = True

    @classmethod
    def check(cls, sql: str) -> dict:
        """对一条 SQL 语句进行安全检查。

        Args:
            sql: 待检查的 SQL 字符串

        Returns:
            {"passed": bool, "errors": [str, ...], "warnings": [str, ...]}
            errors 非空 = 阻断执行
            warnings 非空 = 记录日志但不阻断
        """
        errors = []
        warnings = []

        if not sql or not sql.strip():
            errors.append("SQL 语句为空")
            return {"passed": False, "errors": errors, "warnings": warnings}

        sql_normalized = sql.strip()
        sql_upper = sql_normalized.upper()

        # ---- 1. 只读检查 ----
        first_word = sql_upper.split()[0] if sql_upper.split() else ""
        if first_word == "WITH":
            # CTE 开头，取最后一个 SELECT 之前的部分判断
            pass
        if first_word not in ("SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW"):
            errors.append(
                f"禁止的语句类型: {first_word}。仅允许 SELECT 查询。"
            )

        # ---- 2. 禁止的关键字 ----
        for kw in cls.FORBIDDEN_STATEMENTS:
            pattern = rf"\b{kw}\b"
            if re.search(pattern, sql_upper):
                # 允许在字符串字面量中出现（简单检测）
                if not cls._inside_string_literal(sql, kw):
                    errors.append(
                        f"包含禁止的操作: {kw}。不允许对数据库进行写操作。"
                    )

        # ---- 3. 多语句注入检测 ----
        # 忽略字符串内的分号
        statements_outside_strings = cls._split_outside_strings(sql_normalized, ";")
        if len(statements_outside_strings) > 1:
            non_empty = [s.strip() for s in statements_outside_strings if s.strip()]
            if len(non_empty) > 1:
                errors.append(
                    f"检测到多语句查询 ({len(non_empty)} 条语句)。"
                    f"禁止在一条查询中执行多个 SQL 语句。"
                )

        # ---- 4. 危险函数 ----
        for cmd in cls.DANGEROUS_COMMANDS:
            pattern = rf"\b{cmd}\b"
            if re.search(pattern, sql_upper):
                if not cls._inside_string_literal(sql, cmd):
                    errors.append(
                        f"包含禁止的命令: {cmd}。不允许执行数据库管理操作。"
                    )

        # ---- 5. 代码规范检查 (warnings) ----
        if cls.CODE_QUALITY_CHECKS:
            # SELECT * 警告
            if re.search(r"SELECT\s+\*", sql_upper):
                warnings.append("使用了 SELECT *，建议明确列出需要的字段。")

            # 无 WHERE 子句的 DELETE/UPDATE 已经在上方阻断，这里检查 SELECT 无 LIMIT
            if first_word == "SELECT" and "LIMIT" not in sql_upper:
                if "COUNT(" not in sql_upper and "AVG(" not in sql_upper:
                    pass  # 聚合查询通常不需要 LIMIT

            # 标识符引号检查：建议使用双引号而非无引号
            # （轻量检查：如果字段名包含特殊字符但未加引号）

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    # ---- 辅助方法 ----------------------------------------------------------

    @staticmethod
    def _inside_string_literal(sql, keyword):
        """简单检测关键字是否在字符串字面量内部。"""
        # 找到关键字的所有位置，检查是否在引号内
        sql_lower = sql.lower()
        kw_lower = keyword.lower()
        pos = 0
        while True:
            idx = sql_lower.find(kw_lower, pos)
            if idx == -1:
                return False
            # 检查前面是否有未闭合的单引号
            before = sql[:idx]
            single_quotes = before.count("'") - before.count("''") * 2
            if single_quotes % 2 == 0:
                # 不在字符串内
                return False
            pos = idx + 1
        return True

    @staticmethod
    def _split_outside_strings(sql, delimiter):
        """按分隔符分割 SQL，忽略字符串字面量内的分隔符。"""
        parts = []
        current = []
        in_string = False
        i = 0
        while i < len(sql):
            ch = sql[i]
            if ch == "'" and not in_string:
                in_string = True
                current.append(ch)
            elif ch == "'" and in_string:
                # 检查是否为转义引号 ''
                if i + 1 < len(sql) and sql[i + 1] == "'":
                    current.append("''")
                    i += 1
                else:
                    in_string = False
                    current.append(ch)
            elif ch == delimiter and not in_string:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
            i += 1
        parts.append("".join(current))
        return parts
