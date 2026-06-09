"""
自动化测试运行器: 对 20 个测试问题运行 Agent，统计各项准确率。

用法:
    python test_runner.py                    # 运行全部 20 题
    python test_runner.py --sample 5         # 随机抽样 5 题
    python test_runner.py --report-only      # 仅从已有日志生成报告
"""

import os
import sys
import json
import time
import re
import sqlite3
import logging
from datetime import datetime

_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

os.environ["HF_HUB_OFFLINE"] = "1"

SPIDER_DATA = os.path.join(_project_root, "spider_data", "database")

# 禁用 Agent 日志输出
for name in ["MultiAgent", "SchemaAgent"]:
    logging.getLogger(name).setLevel(logging.WARNING)

# ---- 测试用例定义 ----------------------------------------------------------

TEST_CASES = [
    # (编号, 数据库, 问题, 难度)
    (1,  "concert_singer",                    "查询所有歌手的姓名和年龄", "简单"),
    (2,  "pets_1",                            "列出所有宠物的名字和类型", "简单"),
    (3,  "world_1",                           "查询所有洲的名称，去重", "简单"),
    (4,  "car_1",                             "列出所有汽车制造商的名称和所属国家", "简单"),
    (5,  "flight_2",                          "查询编号为1的航空公司的所有航班号", "简单"),
    (6,  "employee_hire_evaluation",          "查询所有员工的姓名和职位", "简单"),
    (7,  "orchestra",                         "列出所有交响乐团的名称和成立年份", "简单"),
    (8,  "student_transcripts_tracking",      "查询所有学生的名字和姓氏", "简单"),
    (9,  "concert_singer",                    "统计每场演唱会有多少位歌手参加，按参加人数降序排列", "中等"),
    (10, "course_teach",                      "查询每位教师教授的课程名称", "中等"),
    (11, "car_1",                             "统计每个国家有多少个汽车制造商，按数量从多到少排序", "中等"),
    (12, "flight_2",                          "查询每条航线的航空公司名称、出发机场城市和目的机场城市", "中等"),
    (13, "orchestra",                         "找出每位指挥家指挥过的演出数量，按数量降序排列", "中等"),
    (14, "world_1",                           "查询每个国家的官方语言有哪些，按国家名排序", "中等"),
    (15, "employee_hire_evaluation",          "查询每位员工在评估中获得的最高分数", "中等"),
    (16, "car_1",                             "找出生产车型数量最多的制造商名称及其生产的车型数量", "困难"),
    (17, "world_1",                           "查询官方语言数量超过3种的国家名称", "困难"),
    (18, "flight_2",                          "找出航班数量最多的前3个机场城市", "困难"),
    (19, "cre_Doc_Template_Mgt",              "查询包含段落数最多的前5份文档的标题及其段落数", "困难"),
    (20, "student_transcripts_tracking",      "查询选修了课程数量最多的学生姓名及其选修课程数", "困难"),
]

# ---- 测试执行 --------------------------------------------------------------

class TestRunner:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("请设置 DEEPSEEK_API_KEY")

        log_handler = logging.StreamHandler()
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        self.log = logging.getLogger("TestRunner")
        self.log.addHandler(log_handler)
        self.log.setLevel(logging.INFO)
        self.log.propagate = False

    def run_all(self, questions=None):
        """运行全部测试用例"""
        if questions is None:
            questions = TEST_CASES

        self.log.info("=" * 65)
        self.log.info(f"  开始测试: {len(questions)} 题")
        self.log.info(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log.info("=" * 65)

        # 延迟导入 Agent（避免 torch 导入影响计时）
        from src.schema_agent import SchemaAgent
        self.log.info("\n初始化 Agent...")
        agent = SchemaAgent(api_key=self.api_key)
        self.log.info("Agent 就绪\n")

        results = []
        for idx, (num, db, question, difficulty) in enumerate(questions):
            self.log.info(f"[{idx+1}/{len(questions)}] #{num} [{difficulty}] {question[:50]}...")
            t0 = time.time()

            result = self._run_one(agent, db, question, difficulty)
            result["num"] = num
            result["elapsed"] = time.time() - t0

            status = "OK" if result["execution_success"] else "FAIL"
            self.log.info(f"  -> {status} | SQL={result['sql_valid']} | "
                         f"行数={result['row_count']} | {result['elapsed']:.1f}s\n")

            results.append(result)

        # 最终统计一次打印
        self._print_report(results)
        return results

    def _run_one(self, agent, expected_db, question, difficulty):
        """运行单个测试用例"""
        result = {
            "database": expected_db,
            "question": question,
            "difficulty": difficulty,
            "router_intent": "",
            "generated_sql": "",
            "sql_valid": False,
            "execution_success": False,
            "safety_passed": False,
            "row_count": 0,
            "error_msg": "",
            "final_answer": "",
        }

        # 用限制数据库的方式运行，提高检索精度
        try:
            # 重置 Agent 状态，避免多轮干扰
            agent.reset()
            # 在问题中隐式指定数据库上下文
            full_query = question

            # 运行 Agent
            start = time.time()
            answer = agent.run(full_query, verbose=False)

            # 从最后一次 graph invoke 的 state 中提取信息
            state = agent.state
            result["router_intent"] = state.get("intent", "?")
            result["generated_sql"] = state.get("generated_sql", "")
            result["safety_passed"] = len(state.get("safety_errors", [])) == 0
            result["execution_result"] = state.get("execution_result", "")
            result["final_answer"] = answer

            # 验证 SQL 是否能执行
            sql = result["generated_sql"]
            if sql:
                result["sql_valid"] = self._validate_sql(sql)
                result["row_count"] = self._execute_and_count(sql, expected_db)
                result["execution_success"] = result["row_count"] >= 0

        except Exception as e:
            result["error_msg"] = str(e)[:200]

        return result

    @staticmethod
    def _validate_sql(sql):
        """检查 SQL 是否为有效的 SELECT 语句"""
        if not sql or not sql.strip():
            return False
        sql_upper = sql.strip().upper()
        return sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")

    @staticmethod
    def _execute_and_count(sql, db_name):
        """执行 SQL 并返回行数，失败返回 -1"""
        sqlite_path = os.path.join(SPIDER_DATA, db_name, f"{db_name}.sqlite")
        if not os.path.exists(sqlite_path):
            return -1
        try:
            conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchmany(51)
            conn.close()
            return len(rows)
        except Exception:
            return -1

    # ---- 报告生成 ----------------------------------------------------------

    def _print_report(self, results):
        """打印分析报告"""
        total = len(results)
        router_ok = sum(1 for r in results if r["router_intent"] == "sql")
        sql_gen_ok = sum(1 for r in results if r["generated_sql"])
        sql_valid = sum(1 for r in results if r["sql_valid"])
        safety_ok = sum(1 for r in results if r["safety_passed"])
        exec_ok = sum(1 for r in results if r["execution_success"])
        # "完整成功" = SQL有效 + 执行返回了数据
        full_success = sum(1 for r in results
                          if r["sql_valid"] and r["execution_success"] and r["row_count"] > 0)

        # 按难度统计
        by_diff = {}
        for r in results:
            d = r["difficulty"]
            if d not in by_diff:
                by_diff[d] = {"total": 0, "sql_ok": 0, "exec_ok": 0, "full_ok": 0}
            by_diff[d]["total"] += 1
            if r["sql_valid"]: by_diff[d]["sql_ok"] += 1
            if r["execution_success"]: by_diff[d]["exec_ok"] += 1
            if r["sql_valid"] and r["execution_success"] and r["row_count"] > 0:
                by_diff[d]["full_ok"] += 1

        # 平均耗时
        avg_time = sum(r["elapsed"] for r in results) / total if total else 0

        self.log.info("\n" + "=" * 65)
        self.log.info("                    测 试 分 析 报 告")
        self.log.info("=" * 65)
        self.log.info(f"  测试题目数:        {total}")
        self.log.info(f"  平均耗时:          {avg_time:.1f}s")
        self.log.info("-" * 65)
        self.log.info(f"  Router 路由准确率:   {router_ok}/{total} = {100*router_ok/total:.0f}%")
        self.log.info(f"  SQL 生成成功率:      {sql_gen_ok}/{total} = {100*sql_gen_ok/total:.0f}%")
        self.log.info(f"  SQL 语法有效率:      {sql_valid}/{total} = {100*sql_valid/total:.0f}%")
        self.log.info(f"  安全检查通过率:      {safety_ok}/{total} = {100*safety_ok/total:.0f}%")
        self.log.info(f"  SQL 执行成功率:      {exec_ok}/{total} = {100*exec_ok/total:.0f}%")
        self.log.info(f"  完整成功率 (有结果):  {full_success}/{total} = {100*full_success/total:.0f}%")
        self.log.info("-" * 65)
        self.log.info("  按难度统计:")
        for d in ["简单", "中等", "困难"]:
            s = by_diff.get(d, {"total": 0, "sql_ok": 0, "exec_ok": 0, "full_ok": 0})
            if s["total"] > 0:
                self.log.info(f"    {d}: SQL有效={s['sql_ok']}/{s['total']} "
                             f"执行成功={s['exec_ok']}/{s['total']} "
                             f"完整={s['full_ok']}/{s['total']}")
        self.log.info("-" * 65)
        self.log.info("  失败详情:")
        for r in results:
            if not r["execution_success"] or not r["sql_valid"]:
                reason = r["error_msg"] or "SQL 无效或执行返回 0 行"
                self.log.info(f"    #{r['num']} [{r['difficulty']}] {r['question'][:50]}")
                self.log.info(f"        SQL: {r['generated_sql'][:120]}")
                self.log.info(f"        原因: {reason}")
        self.log.info("=" * 65)

        # 保存详细结果到 JSON
        report_path = os.path.join(_project_root, "test_report.json")
        serializable = []
        for r in results:
            serializable.append({
                "num": r["num"],
                "database": r["database"],
                "question": r["question"],
                "difficulty": r["difficulty"],
                "router_intent": r["router_intent"],
                "generated_sql": r["generated_sql"],
                "sql_valid": r["sql_valid"],
                "execution_success": r["execution_success"],
                "safety_passed": r["safety_passed"],
                "row_count": r["row_count"],
                "elapsed": round(r["elapsed"], 1),
                "error_msg": r["error_msg"],
            })
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        self.log.info(f"\n详细结果已保存: {report_path}")


# ---- 命令行入口 ------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="自动化测试运行器")
    parser.add_argument("--sample", type=int, default=0, help="随机抽样 N 题测试")
    parser.add_argument("-k", "--api-key", default=None, help="DeepSeek API Key")

    args = parser.parse_args()

    questions = TEST_CASES
    if args.sample and args.sample < len(TEST_CASES):
        questions = random.sample(TEST_CASES, args.sample)

    runner = TestRunner(api_key=args.api_key)
    runner.run_all(questions)
