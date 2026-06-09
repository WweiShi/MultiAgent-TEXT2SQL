"""
图表生成工具：根据 SQL 查询结果绘制饼状图、柱状图、折线图。

用法:
    from src.chart_tool import create_chart
    result = create_chart("bar", data_text, title="部门薪资对比")
"""

import os
import re
import hashlib
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 项目根目录，图表输出到 output/charts/
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHART_OUTPUT = os.path.join(PROJECT_DIR, "output", "charts")
os.makedirs(CHART_OUTPUT, exist_ok=True)

# ---- 中文字体设置 ----------------------------------------------------------

def _setup_chinese_font():
    """尝试配置中文字体，按优先级查找。"""
    candidates = [
        "Microsoft YaHei", "SimHei", "KaiTi", "FangSong",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font
    # fallback: 不设中文字体，至少图表能生成
    return None

_FONT = _setup_chinese_font()

# ---- 公共 API --------------------------------------------------------------

def create_chart(chart_type: str, data_text: str, title: str = "",
                 x_label: str = "", y_label: str = "") -> str:
    """根据数据文本生成图表并保存为 PNG 文件。

    Args:
        chart_type: "pie" | "bar" | "line"
        data_text: submit_sql 返回的表格文本，格式为:
                    列1 | 列2
                    -------------
                    val1 | val2
                    ...
        title: 图表标题
        x_label: X 轴标签（饼图忽略）
        y_label: Y 轴标签（饼图忽略）

    Returns:
        成功: "图表已生成: output/charts/xxx.png"
        失败: "图表生成失败: 错误信息"
    """
    try:
        labels, values, col_names = _parse_table_data(data_text)
        if not labels or not values:
            return "图表生成失败: 无法从数据中解析出标签和数值列。请确保查询结果至少包含一列文本和一列数字。"

        chart_type = chart_type.lower().strip()
        if chart_type not in ("pie", "bar", "line"):
            return f"图表生成失败: 不支持的图表类型 '{chart_type}'，支持: pie, bar, line"

        # 限制数据点数量，避免图表过于拥挤
        if len(labels) > 20:
            labels = labels[:20]
            values = values[:20]
            title += " (Top 20)"

        # 生成图表
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "pie":
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 9}
            )
            ax.set_title(title or "饼状图", fontsize=14, fontweight="bold")
        elif chart_type == "bar":
            colors = plt.cm.Set3.colors[:len(labels)] if len(labels) <= 12 else plt.cm.viridis(
                [i / len(labels) for i in range(len(labels))]
            )
            bars = ax.bar(labels, values, color=colors)
            ax.set_xlabel(x_label or col_names[0])
            ax.set_ylabel(y_label or col_names[1])
            ax.set_title(title or "柱状图", fontsize=14, fontweight="bold")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            # 在柱子上标注数值
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                        str(val), ha="center", va="bottom", fontsize=8)
        elif chart_type == "line":
            ax.plot(labels, values, marker="o", linewidth=2, markersize=6)
            ax.set_xlabel(x_label or col_names[0])
            ax.set_ylabel(y_label or col_names[1])
            ax.set_title(title or "折线图", fontsize=14, fontweight="bold")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(f"{title}{chart_type}{timestamp}".encode()).hexdigest()[:8]
        filename = f"{chart_type}_{name_hash}_{timestamp}.png"
        filepath = os.path.join(CHART_OUTPUT, filename)

        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 返回相对路径
        rel_path = os.path.relpath(filepath, PROJECT_DIR).replace("\\", "/")
        return f"图表已生成: {rel_path}"

    except Exception as e:
        import traceback
        return f"图表生成失败: {e}"


# ---- 数据解析 --------------------------------------------------------------

def _parse_table_data(data_text: str) -> tuple:
    """从 submit_sql 的输出文本中解析出标签列和数值列。

    输入格式:
        col1 | col2
        -------------
        val1 | val2
        ...

    返回: (labels, values, col_names)
    """
    lines = data_text.strip().split("\n")

    # 跳过行数标记 "(N 行)"
    if lines and lines[0].startswith("(") and "行)" in lines[0]:
        lines = lines[1:]

    if len(lines) < 2:
        return [], [], []

    # 第一行是列名
    header = lines[0]
    col_names = [c.strip() for c in header.split("|")]

    # 跳过分隔线 (含 '---' 的行)
    data_lines = []
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("-") or line.startswith("..."):
            continue
        parts = [c.strip() for c in line.split("|")]
        if len(parts) >= 2:
            data_lines.append(parts)

    if not data_lines:
        return [], [], col_names

    # 第一列作为标签，搜索第一个数值列作为值
    labels = [d[0] for d in data_lines]

    # 在剩余列中找到最佳数值列（非空、可转为数字的列）
    values = None
    value_col_idx = 1
    for col_idx in range(1, len(data_lines[0])):
        try:
            candidate = []
            for d in data_lines:
                val = d[col_idx].replace(",", "").replace("%", "").replace("$", "").strip()
                if val:
                    candidate.append(float(val))
                else:
                    candidate.append(0.0)
            if any(v != 0 for v in candidate):
                values = candidate
                value_col_idx = col_idx
                break
        except (ValueError, IndexError):
            continue

    if values is None:
        return [], [], col_names

    # 如果标签也是数字（比如年份），排序
    return labels, values, col_names
