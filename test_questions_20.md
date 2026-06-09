# Text-to-SQL 测试问题集 (20 题)

## 简单 (8 题) — 单表查询、基础过滤、简单聚合

1. 【concert_singer】查询所有歌手的姓名和年龄。
2. 【pets_1】列出所有宠物的名字和类型（PetType）。
3. 【world_1】查询所有洲（Continent）的名称，去重。
4. 【car_1】列出所有汽车制造商的名称（FullName）和所属国家（Country）。
5. 【flight_2】查询编号为 1 的航空公司（Airline）的所有航班号（FlightNo）。
6. 【employee_hire_evaluation】查询所有员工的姓名（Name）和职位（Position）。
7. 【orchestra】列出所有交响乐团的名称（Orchestra）和成立年份（Year_of_Founded）。
8. 【student_transcripts_tracking】查询所有学生的名字（first_name）和姓氏（last_name）。

## 中等 (7 题) — 多表 JOIN、GROUP BY、排序

9. 【concert_singer】统计每场演唱会（concert_Name）有多少位歌手参加，按参加人数降序排列。
10. 【course_teach】查询每位教师（Name）教授的课程名称（Course_Name）。
11. 【car_1】统计每个国家（Country）有多少个汽车制造商，按数量从多到少排序。
12. 【flight_2】查询每条航线的航空公司名称（Airline）、出发机场城市（SourceAirport）和目的机场城市（DestAirport）。
13. 【orchestra】找出每位指挥家（Name）指挥过的演出（Performance）数量，按数量降序排列。
14. 【world_1】查询每个国家（Name）的官方语言（Language）有哪些，按国家名排序。
15. 【employee_hire_evaluation】查询每位员工（Name）在评估（Evaluation）中获得的最高分数（Score）。

## 困难 (5 题) — 子查询、HAVING、复杂聚合

16. 【car_1】找出生产车型数量最多的制造商名称（Maker）及其生产的车型数量。
17. 【world_1】查询官方语言数量超过 3 种的国家名称。
18. 【flight_2】找出每天航班数量最多的前 3 个机场城市（City）。
19. 【cre_Doc_Template_Mgt】查询包含段落数最多的前 5 份文档的标题（Document_Title）及其段落数。
20. 【student_transcripts_tracking】查询选修了最多课程的学生姓名（first_name + last_name）及其选修课程数。
