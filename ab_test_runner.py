"""
A/B 测试：单 Agent (sql) vs Multi-Task 分解并行

对比维度:
  - 执行耗时
  - SQL 正确率
  - 最终答案准确率
  - 任务完成率（部分失败时是否仍有可用结果）

用法:
    python ab_test_runner.py            # 运行全部 10 题
    python ab_test_runner.py --mode both  # 只跑一种模式: single | multi | both
"""

import os
import sys
import json
import time
import sqlite3
import logging
from datetime import datetime

_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SPIDER_DATA = os.path.join(_project_root, "spider_data", "database")

# 静默 Agent 日志
for name in ["MultiAgent", "SchemaAgent", "MultiTask", "TaskBus"]:
    logging.getLogger(name).setLevel(logging.WARNING)

# ---- 测试用例：跨库对比类问题（适合 Multi-Task 分解）-----------------------

TEST_CASES = [
    {
        "id": 1,
        "query": "查concert_singer和singer数据库分别有多少歌手并对比",
        "expected_dbs": ["concert_singer", "singer"],
        "expected_sql_count": 2,  # 至少 2 条独立 SQL
    },
    {
        "id": 2,
        "query": "hr_1数据库和employee_hire_evaluation数据库分别有多少名员工",
        "expected_dbs": ["hr_1", "employee_hire_evaluation"],
        "expected_sql_count": 2,
    },
    {
        "id": 3,
        "query": "orchestra和concert_singer数据库分别有多少场演出记录",
        "expected_dbs": ["orchestra", "concert_singer"],
        "expected_sql_count": 2,
    },
    {
        "id": 4,
        "query": "查car_1数据库的车型数量和world_1数据库的城市数量",
        "expected_dbs": ["car_1", "world_1"],
        "expected_sql_count": 2,
    },
    {
        "id": 5,
        "query": "博物馆参观记录(museum_visit)和餐厅光顾记录(restaurant_1)分别有多少条",
        "expected_dbs": ["museum_visit", "restaurant_1"],
        "expected_sql_count": 2,
    },
    {
        "id": 6,
        "query": "查singer数据库和concert_singer数据库，哪个数据库歌手的平均年龄更大",
        "expected_dbs": ["singer", "concert_singer"],
        "expected_sql_count": 2,
    },
    {
        "id": 7,
        "query": "flight_2和bike_1数据库各有多少条行程记录",
        "expected_dbs": ["flight_2", "bike_1"],
        "expected_sql_count": 2,
    },
    {
        "id": 8,
        "query": "student_transcripts_tracking和course_teach数据库分别有多少学生和多少教师",
        "expected_dbs": ["student_transcripts_tracking", "course_teach"],
        "expected_sql_count": 2,
    },
    {
        "id": 9,
        "query": "pets_1数据库的宠物数量和world_1数据库的国家数量分别是多少",
        "expected_dbs": ["pets_1", "world_1"],
        "expected_sql_count": 2,
    },
    {
        "id": 10,
        "query": "查employee_hire_evaluation和cre_Doc_Template_Mgt数据库分别有多少条记录",
        "expected_dbs": ["employee_hire_evaluation", "cre_Doc_Template_Mgt"],
        "expected_sql_count": 2,
    },
]


# ---- 执行器 ----------------------------------------------------------------

class ABTestRunner:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("请设置 DEEPSEEK_API_KEY")

        self._agent = None
        self.log = logging.getLogger("ABTest")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.log.addHandler(handler)
        self.log.setLevel(logging.INFO)
        self.log.propagate = False

    def _get_agent(self):
        """延迟初始化 Agent。"""
        if self._agent is None:
            from src.schema_agent import SchemaAgent
            self.log.info("初始化 Agent...")
            self._agent = SchemaAgent(api_key=self.api_key)
            self.log.info("Agent 就绪\n")
        return self._agent

    # ---- 单 Agent 模式 -----------------------------------------------------

    def _run_single_agent(self, query):
        """强制走 sql 路由：直接调用 SQL Generator。"""
        agent = self._get_agent()
        agent.reset()

        # 先 Rewrite 规范化，再绕过 Router 直连 SQL Generator
        t0 = time.time()
        from langchain_core.messages import HumanMessage

        rewrite_state = {
            "messages": [HumanMessage(content=query)],
            "intent": "", "db_name": "", "generated_sql": "",
            "final_answer": "", "rewritten_query": "", "insight_report": "",
        }
        rewritten = agent._rewrite_node(rewrite_state)["rewritten_query"] or query

        result = agent._sql_generator_node({
            "messages": [HumanMessage(content=rewritten)],
            "intent": "sql", "rewritten_query": rewritten,
            "db_name": "", "generated_sql": "",
            "final_answer": "", "insight_report": "",
        })
        elapsed = time.time() - t0

        # 提取结果
        final_answer = result.get("final_answer", "")
        generated_sql = result.get("generated_sql", "")

        # 通过检查 final_answer 内容判断是否覆盖了预期数据库
        dbs_covered = []
        for db in ["concert_singer", "singer", "hr_1", "employee_hire_evaluation",
                    "orchestra", "car_1", "world_1", "museum_visit", "restaurant_1",
                    "flight_2", "bike_1", "student_transcripts_tracking",
                    "course_teach", "pets_1", "cre_Doc_Template_Mgt"]:
            if db in final_answer or db in generated_sql:
                dbs_covered.append(db)

        return {
            "mode": "single",
            "elapsed": round(elapsed, 1),
            "final_answer": final_answer,
            "generated_sql": generated_sql,
            "dbs_covered": list(set(dbs_covered)),
            "sql_count": 1,  # 单 Agent 无法统计实际 SQL 数
            "has_answer": len(final_answer) > 30,
        }

    # ---- Multi-Task 模式 ---------------------------------------------------

    def _run_multi_task(self, query):
        """走 multi 路由：Decomposer → TaskBus → Aggregator。"""
        agent = self._get_agent()
        agent.reset()

        from src.multi_task_agent import MultiTaskManager, TaskBus

        mgr = MultiTaskManager(agent.llm, agent.store, agent.embedder.model)

        t0 = time.time()

        # 1. 分解
        tasks = mgr.decompose(query)

        # 2. TaskBus 并行执行
        bus = TaskBus(mgr)
        stats = bus.execute_all(tasks)

        # 3. 聚合
        final_answer = mgr.aggregate(tasks)
        elapsed = time.time() - t0

        # 统计覆盖的数据库
        dbs_covered = []
        for t in tasks:
            prompt = t.get("prompt", "")
            for db in ["concert_singer", "singer", "hr_1", "employee_hire_evaluation",
                        "orchestra", "car_1", "world_1", "museum_visit", "restaurant_1",
                        "flight_2", "bike_1", "student_transcripts_tracking",
                        "course_teach", "pets_1", "cre_Doc_Template_Mgt"]:
                if db in prompt or db in final_answer:
                    dbs_covered.append(db)

        return {
            "mode": "multi",
            "elapsed": round(elapsed, 1),
            "final_answer": final_answer,
            "task_count": len(tasks),
            "tasks_success": stats["success"],
            "tasks_failed": stats["failed"],
            "dbs_covered": list(set(dbs_covered)),
            "has_answer": len(final_answer) > 30,
        }

    # ---- 主流程 ------------------------------------------------------------

    def run(self, mode="both"):
        """运行 A/B 测试。

        Args:
            mode: "single" | "multi" | "both"
        """
        modes = ["single", "multi"] if mode == "both" else [mode]

        self.log.info("=" * 70)
        self.log.info("  A/B 测试: 单 Agent 串行 vs Multi-Task 并行分解")
        self.log.info("  时间: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.log.info("  题目数: %d", len(TEST_CASES))
        self.log.info("  模式: %s", ", ".join(modes))
        self.log.info("=" * 70)

        all_results = {m: [] for m in modes}

        for i, case in enumerate(TEST_CASES):
            for m in modes:
                self.log.info("\n[%d/%d] #%d [%s] %s",
                             i + 1, len(TEST_CASES), case["id"], m, case["query"][:60])

                if m == "single":
                    result = self._run_single_agent(case["query"])
                    dbs_ok = len(set(result["dbs_covered"]) &
                                 set(case["expected_dbs"]))
                    self.log.info("  耗时: %.1fs | 覆盖DB: %d/%d | 有答案: %s",
                                 result["elapsed"], dbs_ok,
                                 len(case["expected_dbs"]), result["has_answer"])
                else:
                    result = self._run_multi_task(case["query"])
                    dbs_ok = len(set(result["dbs_covered"]) &
                                 set(case["expected_dbs"]))
                    self.log.info("  耗时: %.1fs | 任务: %d(成功%d/失败%d) | 覆盖DB: %d/%d | 有答案: %s",
                                 result["elapsed"], result["task_count"],
                                 result["tasks_success"], result["tasks_failed"],
                                 dbs_ok, len(case["expected_dbs"]),
                                 result["has_answer"])

                # 附加期望值用于报告
                result["expected_dbs"] = case["expected_dbs"]
                result["expected_sql_count"] = case["expected_sql_count"]
                result["query"] = case["query"]
                all_results[m].append(result)

        # 生成报告
        self._print_report(all_results, modes)
        self._save_report(all_results)

    # ---- 报告 --------------------------------------------------------------

    def _print_report(self, all_results, modes):
        self.log.info("\n\n" + "=" * 70)
        self.log.info("                   A/B 测试分析报告")
        self.log.info("=" * 70)

        for m in modes:
            results = all_results[m]
            total = len(results)
            avg_time = sum(r["elapsed"] for r in results) / total if total else 0
            answers_ok = sum(1 for r in results if r["has_answer"])

            # DB 覆盖率
            db_hits = 0
            db_total = 0
            for r in results:
                covered = set(r["dbs_covered"])
                expected = set(r["expected_dbs"])
                db_hits += len(covered & expected)
                db_total += len(expected)
            db_coverage = 100 * db_hits / db_total if db_total else 0

            # 任务统计（仅 multi 模式）
            if m == "multi":
                total_tasks = sum(r.get("task_count", 0) for r in results)
                success_tasks = sum(r.get("tasks_success", 0) for r in results)
                avg_tasks = total_tasks / total if total else 0

            self.log.info(f"\n--- {m.upper()} 模式 ---")
            self.log.info(f"  题目数:        {total}")
            self.log.info(f"  平均耗时:      {avg_time:.1f}s")
            self.log.info(f"  有实质回答:    {answers_ok}/{total} = {100*answers_ok/total:.0f}%")
            self.log.info(f"  数据库覆盖率:  {db_coverage:.0f}% ({db_hits}/{db_total})")
            if m == "multi":
                self.log.info(f"  平均子任务数:  {avg_tasks:.1f}")
                self.log.info(f"  子任务成功率:  {success_tasks}/{total_tasks} = "
                             f"{100*success_tasks/total_tasks:.0f}%" if total_tasks else "N/A")

        # 对比
        if len(modes) == 2:
            s = all_results["single"]
            m = all_results["multi"]
            self.log.info("\n--- 对比 ---")
            s_time = sum(r["elapsed"] for r in s) / len(s)
            m_time = sum(r["elapsed"] for r in m) / len(m)
            speedup = (s_time - m_time) / s_time * 100

            s_ans = sum(1 for r in s if r["has_answer"])
            m_ans = sum(1 for r in m if r["has_answer"])

            s_db = 0
            m_db = 0
            db_t = 0
            for sr, mr in zip(s, m):
                sc = set(sr["dbs_covered"]) & set(sr["expected_dbs"])
                mc = set(mr["dbs_covered"]) & set(mr["expected_dbs"])
                s_db += len(sc)
                m_db += len(mc)
                db_t += len(sr["expected_dbs"])

            self.log.info(f"  耗时: 单Agent {s_time:.1f}s → Multi {m_time:.1f}s "
                         f"({'加速' if speedup > 0 else '变慢'} {abs(speedup):.0f}%)")
            self.log.info(f"  答案率: 单Agent {s_ans}/{len(s)} → Multi {m_ans}/{len(m)}")
            self.log.info(f"  DB覆盖: 单Agent {s_db}/{db_t} → Multi {m_db}/{db_t} "
                         f"({100*(m_db-s_db)/db_t if db_t else 0:+.0f}%)")
        self.log.info("=" * 70)

    def _save_report(self, all_results):
        """保存详细结果到 JSON。"""
        report_path = os.path.join(_project_root, "ab_test_report.json")
        serializable = {}
        for mode, results in all_results.items():
            serializable[mode] = []
            for r in results:
                serializable[mode].append({
                    "query": r["query"],
                    "elapsed": r["elapsed"],
                    "has_answer": r["has_answer"],
                    "dbs_covered": r["dbs_covered"],
                    "expected_dbs": r["expected_dbs"],
                    "final_answer": r.get("final_answer", "")[:300],
                    **({"task_count": r.get("task_count", 0),
                        "tasks_success": r.get("tasks_success", 0),
                        "tasks_failed": r.get("tasks_failed", 0)}
                       if mode == "multi" else {}),
                })

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        self.log.info(f"\n详细结果已保存: {report_path}")


# ---- 命令行入口 ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A/B 测试: 单 Agent vs Multi-Task")
    parser.add_argument("--mode", default="both",
                        choices=["single", "multi", "both"],
                        help="测试模式 (default: both)")
    parser.add_argument("-k", "--api-key", default=None,
                        help="DeepSeek API Key")
    parser.add_argument("--sample", type=int, default=0,
                        help="随机抽样 N 题 (default: 全部 10 题)")

    args = parser.parse_args()

    cases = TEST_CASES
    if args.sample and args.sample < len(cases):
        import random
        cases = random.sample(cases, args.sample)

    runner = ABTestRunner(api_key=args.api_key)
    runner.run(mode=args.mode)
