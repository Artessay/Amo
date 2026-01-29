import json
import matplotlib.pyplot as plt

# 设置中文字体和科研论文风格
plt.rcParams.update({
    # 'font.family': 'serif',
    # 'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

# 文件路径
files = [
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_lag1.json",
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_lag3.json",
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_lag5.json",
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_lag7.json"
]

# 提取lag值
lags = [1, 3, 5, 7]

# 存储数据
accuracy = []
conciseness = []
format = []

# 读取数据
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
    metrics = data["DigitalLearningGmbH/MATH-lighteval"]
    accuracy.append(metrics["math_accuracy"])
    conciseness.append(metrics["math_conciseness"])
    format.append(metrics["math_format"])

# 绘制折线图
fig, ax = plt.subplots()

# 绘制三条折线
ax.plot(lags, accuracy, marker='o', label='Accuracy', color='blue')
ax.plot(lags, conciseness, marker='s', label='Conciseness', color='green')
ax.plot(lags, format, marker='^', label='Format', color='red')

# 设置标题和标签
# ax.set_title('Performance Metrics vs. Lag Value', fontweight='bold')
ax.set_xlabel('Lag Value')
ax.set_ylabel('Score')

# 设置x轴刻度
ax.set_xticks(lags)
ax.set_xticklabels([f'{lag}' for lag in lags])

# 设置y轴范围
ax.set_ylim(0.7, 1.0)

# 添加图例
ax.legend(loc='lower right')

# 添加数值标签
for i, lag in enumerate(lags):
    ax.text(lag, accuracy[i] + 0.005, f'{accuracy[i]:.3f}', ha='center', va='bottom', fontsize=9)
    ax.text(lag, conciseness[i] + 0.005, f'{conciseness[i]:.3f}', ha='center', va='bottom', fontsize=9)
    ax.text(lag, format[i] + 0.005, f'{format[i]:.3f}', ha='center', va='bottom', fontsize=9)

# 调整布局
plt.tight_layout()

# 保存为PDF和PNG格式
plt.savefig('./playground/visualization/lag_analysis.pdf', format='pdf', bbox_inches='tight')
plt.savefig('./playground/visualization/lag_analysis.png', format='png', bbox_inches='tight')

# 显示图表
plt.show()

# 打印数据
print("Lag Analysis Results:")
print("Lag | Accuracy | Conciseness | Format")
print("-" * 50)
for i, lag in enumerate(lags):
    print(f"{lag:3d} | {accuracy[i]:.4f}    | {conciseness[i]:.4f}     | {format[i]:.4f}")
