import json
import numpy as np
import matplotlib.pyplot as plt
# ======================
# 全局风格（论文级）
# ======================
# plt.rc('font', family='Times New Roman')
plt.rcParams.update({
    "figure.dpi": 300
})
# ======================
# 文件路径 & lag
# ======================
files = [
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_lag1.json",
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_lag3.json",
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_lag5.json",
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_lag7.json"
]
lags = np.array([1, 3, 5, 7])
accuracy, conciseness, format_score = [], [], []
# ======================
# 读取数据（老方法）
# ======================
for file in files:
    with open(file, "r") as f:
        data = json.load(f)
    metrics = data["DigitalLearningGmbH/MATH-lighteval"]
    accuracy.append(metrics["math_accuracy"])
    conciseness.append(metrics["math_conciseness"])
    format_score.append(metrics["math_format"])
accuracy     = np.array(accuracy)
conciseness  = np.array(conciseness)
format_score = np.array(format_score)
# overall = (accuracy + conciseness + format_score) / 3
overall = (accuracy * conciseness * format_score) 
print("overall", overall)
# ======================
# 指标配置
# ======================
metrics = [
    ("Correctness", accuracy,     (0.73, 0.79), "#023047"),
    ("Conciseness", conciseness, (0.935, 0.975), "#E76F51"),
    ("Clarity", format_score,   (0.97, 0.995), "#8AB17D"),
    # ("Overall", overall,       (0.88, 0.91), "#6A4C93"), # 新增 Overall 图表
    ("Hypervolume", overall,       (0.65, 0.75), "#6A4C93"), # 新增 Overall 图表
]

# ======================
# 画图
# ======================
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)  # 2行2列，拼成4个子图
axes = axes.flatten()   # 方便统一遍历
for ax, (name, values, ylim, color) in zip(axes, metrics):
    ax.grid(linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 坐标轴线加粗
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.plot(
        lags, values,
        marker="o",
        linewidth=3.5,
        markersize=14,
        color=color,
        zorder=3
    )
    # --- 动态标签避让逻辑 ---
    y_range = ylim[1] - ylim[0]
    for i in range(len(values)):
        x, y = lags[i], values[i]
        # 判断是否为局部最小值或处于下降趋势
        is_bottom = False
        if i == 0:
            if values[i] < values[i+1]: is_bottom = True
        elif i == len(values) - 1:
            if values[i] < values[i-1]: is_bottom = True
        else:
            if values[i] <= values[i-1] and values[i] <= values[i+1]: is_bottom = True
        # 动态调整偏移
        offset = -(y_range * 0.1) if is_bottom else (y_range * 0.05)
        va = "top" if is_bottom else "bottom"
        ax.text(
            x, y + offset,
            f"{y:.3f}",
            ha="center",
            va=va,
            fontsize=20,
            fontweight="bold",
            color=color
        )
    ax.set_title(name, fontsize=26, fontweight="bold", pad=15)
    ax.set_ylim(*ylim)
    ax.set_xticks(lags)
    ax.set_xlim(lags[0] - 1, lags[-1] + 1)
    # --- 坐标轴刻度：加粗、变大 ---
    ax.tick_params(axis="both", labelsize=22, width=2, length=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    # 移除横坐标标签，只保留刻度数字（你可以去掉 set_xlabel，也可只置空）
    ax.set_xlabel("")    # 不设置"x label"

# 移除全局 x label（"Lag Value"）
# plt.subplots_adjust 设置
plt.subplots_adjust(
    left=0.07,
    right=0.98,
    bottom=0.09,
    top=0.92,
    wspace=0.30,
    hspace=0.28      # 新增 hspace
)

# 保存
plt.savefig("./playground/visualization/lag_analysis.pdf", bbox_inches="tight")
plt.savefig("./playground/visualization/lag_analysis.png", bbox_inches="tight")
plt.show()