import json

# 文件路径
files = [
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_distance.json",
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo_euclidean.json",
    "./results/MATH-lighteval/qwen2.5-1.5b_hvpo.json",
]

# 提取lag值
lags = ["Constant Zero", "Euclidean Distance", "Chebyshve Distance"]

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


# 打印数据
print("Lag Analysis Results:")
print("Distance Metric    | Correctness | Conciseness | Clarity   | Hypervolume")
print("-" * 75)
for i, lag in enumerate(lags):
    hypervolume = accuracy[i] * conciseness[i] * format[i]
    print(f"{lag:18s} | {accuracy[i]:.6f}    | {conciseness[i]:.5f}     | {format[i]:.4f}    | {hypervolume:.4f}")
