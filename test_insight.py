import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['DEEPSEEK_API_KEY'] = 'sk-fa02696716a6441596a2657034f35bd8'

from src.schema_agent import SchemaAgent
agent = SchemaAgent()

# 测试 1
print("=" * 55)
print("测试 1: 分析 hr_1 的员工薪资分布")
a = agent.run("hr_1 的员工薪资分布是怎样的", verbose=False)
if a: print(a[:800])

# 测试 2
agent.reset()
print()
print("=" * 55)
print("测试 2: bike_1 的行程数据有什么特点")
a = agent.run("bike_1 的行程数据有什么特点", verbose=False)
if a: print(a[:600])
