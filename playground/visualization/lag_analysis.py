import json

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


# 打印数据
print("Lag Analysis Results:")
print("Lag | Correctness | Conciseness | Clarity     | Hypervolume")
print("-" * 50)
for i, lag in enumerate(lags):
    hypervolume = accuracy[i] * conciseness[i] * format[i]
    print(f"{lag:3d} | {accuracy[i]:.4f}    | {conciseness[i]:.4f}     | {format[i]:.4f}    | {hypervolume:.4f}")
