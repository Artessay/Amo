import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects # 用于给文字加描边

# --- 1. 全局绘图风格设置 ---
# 使用 seaborn-v0_8-paper 风格作为基础（如果已安装），或者手动定义
plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    'font.family': 'sans-serif',        # 优先使用无衬线字体，更现代
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'SimHei'], # 兼容中文
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.figsize': (8, 8),           # 调整为正方形，适合雷达图
    'figure.dpi': 300,
    'lines.linewidth': 2.5,             # 线条稍粗，突出数据
    'axes.grid': True,
    'grid.color': '#DDDDDD',            # 网格颜色更淡
    'grid.linestyle': '--',             # 网格设为虚线
    'grid.linewidth': 0.8,
})

# --- 2. 数据准备 ---
# 开关：如果没有真实文件，设为 True 使用模拟数据运行查看效果
USE_MOCK_DATA = False 

# 文件路径 & Lag值
files = [
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_lag1.json",
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_lag3.json",
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_lag5.json",
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_lag7.json"
]
lags = [1, 3, 5, 7]
categories = ['Accuracy', 'Conciseness', 'Format', 'Overall']
n_categories = len(categories)

accuracy, conciseness, format_score, overall = [], [], [], []

if USE_MOCK_DATA:
    # 模拟数据 (仅用于演示效果)
    print("Warning: Using Mock Data for demonstration.")
    accuracy = [0.85, 0.88, 0.90, 0.92]
    conciseness = [0.90, 0.85, 0.88, 0.95]
    format_score = [0.95, 0.96, 0.94, 0.98]
    for i in range(4):
        overall.append((accuracy[i] + conciseness[i] + format_score[i]) / 3)
else:
    # 读取真实数据
    try:
        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
            metrics = data["DigitalLearningGmbH/MATH-lighteval"]
            acc = metrics["math_accuracy"]
            conc = metrics["math_conciseness"]
            fmt = metrics["math_format"]
            accuracy.append(acc)
            conciseness.append(conc)
            format_score.append(fmt)
            overall.append((acc + conc + fmt) / 3)
    except FileNotFoundError:
        print("Error: JSON files not found. Please check paths or set USE_MOCK_DATA = True.")
        exit()

# --- 3. 绘图逻辑 ---

# 角度计算 (保持闭合)
angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
angles += angles[:1] 

# 创建画布
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

# 调整极坐标的起始方向（0度在正上方）和方向（顺时针）
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 设置更美观的科研配色 (Hex codes)
# 依次为: 深蓝, 蓝绿, 橙色, 砖红
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'] 

# 绘制每个lag值
for i, lag in enumerate(lags):
    values = [accuracy[i], conciseness[i], format_score[i], overall[i]]
    values += values[:1] # 闭合数据
    
    color = colors[i % len(colors)]
    
    # 绘制线条
    ax.plot(angles, values, linewidth=2, linestyle='-', label=f'Lag {lag}', color=color, marker='o', markersize=5)
    
    # 绘制填充 (透明度更低，避免重叠混浊)
    ax.fill(angles, values, alpha=0.1, color=color)

    # 优化数值标签
    # 仅在非重叠严重的情况下显示，或者只显示最高/最低 Lag 的数值
    # 这里演示给每个点加标签，但使用了 PathEffects 增加白色描边，使其在网格线上也清晰可见
    for angle, value in zip(angles[:-1], values[:-1]):
        # 根据角度计算文本偏移，避免文字压住线条
        # 简单策略：数值稍微向外偏一点
        txt = ax.text(angle, value + 0.015, f'{value:.2f}', 
                      ha='center', va='center', fontsize=9, color=color, fontweight='bold')
        # 给文字加白色描边，使其在复杂背景下更清晰
        txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# --- 4. 坐标轴美化 ---

# 设置X轴（类别标签）
ax.set_xticks(angles[:-1])
# 增加标签与轴的距离 (pad)
ax.set_xticklabels(categories, fontsize=13, fontweight='bold', color='#333333')
ax.tick_params(axis='x', pad=15) 

# 设置Y轴（数值刻度）
ax.set_rlabel_position(0) # 将Y轴刻度标签放在0度方向
# 动态调整Y轴范围，避免数据挤在一起
y_min = min(min(accuracy), min(conciseness), min(format_score), min(overall)) - 0.05
y_max = max(max(accuracy), max(conciseness), max(format_score), max(overall)) + 0.05
# 如果数据范围都在 0.7-1.0 之间，可以手动锁定让差异更明显
# ax.set_ylim(0.7, 1.0) 
ax.set_ylim(max(0, y_min), min(1.05, y_max)) # 自动适应，但保持下限不小于0

plt.yticks(fontsize=9, color='gray') # Y轴刻度字体变灰，减少视觉干扰
ax.grid(True, linestyle='--', alpha=0.6, color='gray') # 网格线优化

# 去除外圈粗边框，看起来更现代
ax.spines['polar'].set_visible(False)

# --- 5. 图例与布局 ---

# 图例放在底部，水平排列
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          ncol=4, frameon=False, fontsize=12)

# 添加标题
plt.title("Performance Metrics Comparison across Lags", y=1.08, fontsize=16, fontweight='bold', color='#222222')

plt.tight_layout()

# 保存
plt.savefig('./radar_analysis.pdf', format='pdf', bbox_inches='tight')
plt.savefig('./radar_analysis.png', format='png', bbox_inches='tight', dpi=300)

plt.show()

# --- 6. 打印数据表格 ---
print("\n=== Detailed Results ===")
print(f"{'Lag':<5} | {'Accuracy':<10} | {'Conciseness':<12} | {'Format':<10} | {'Overall':<10}")
print("-" * 55)
for i, lag in enumerate(lags):
    print(f"{lag:<5d} | {accuracy[i]:<10.4f} | {conciseness[i]:<12.4f} | {format_score[i]:<10.4f} | {overall[i]:<10.4f}")