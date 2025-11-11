#!/bin/bash
set -e  # 报错立即停止脚本

##############################################################################
# 第一步：配置全局参数（甲方可根据自身环境修改以下变量）
##############################################################################
# 基础模型路径（可替换为Alpaca衍生版，如 chavinlo/alpaca-native）
BASE_MODEL="llava-hf/llava-1.5-7b-hf"
# 数据集路径（需确保甲方本地有该文件夹及对应的JSONL文件）
DATA_DIR="./data"
# 模型保存路径
HELPFUL_MODEL_OUTPUT="./output/helpful_reward_model"
HARMLESS_MODEL_OUTPUT="./output/harmless_reward_model"
# 日志保存路径
LOG_DIR="./logs"
# 训练配置文件路径
HELPFUL_CONFIG="./config_helpful.yaml"
HARMLESS_CONFIG="./config_harmless.yaml"
# 硬件配置（自动适配GPU数量，无需修改）
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
echo "=== 检测到 $GPU_NUM 块GPU，将使用DeepSpeed分布式训练 ==="

##############################################################################
# 第二步：环境配置（自动安装依赖，无需手动操作）
##############################################################################
echo -e "\n=== 开始配置训练环境 ==="
# 检查Python版本（要求3.8+）
PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1-2)
if [[ $(echo "$PYTHON_VERSION < 3.8" | bc) -eq 1 ]]; then
    echo "错误：Python版本需≥3.8，当前版本为 $PYTHON_VERSION"
    exit 1
fi

# 安装依赖
echo "=== 安装必要依赖包 ==="
pip3 install --upgrade pip
pip3 install -r requirements.txt > install.log 2>&1 || { echo "依赖安装失败，查看install.log"; exit 1; }
echo "=== 环境配置完成 ==="

##############################################################################
# 第三步：数据校验（确保数据集存在，避免训练报错）
##############################################################################
echo -e "\n=== 开始校验数据集 ==="
REQUIRED_FILES=(
    "$DATA_DIR/helpful_train.jsonl"
    "$DATA_DIR/helpful_eval.jsonl"
    "$DATA_DIR/harmless_train.jsonl"
    "$DATA_DIR/harmless_eval.jsonl"
)
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "错误：数据集文件 $file 不存在，请检查路径"
        exit 1
    fi
done
echo "=== 数据集校验通过 ==="

##############################################################################
# 第四步：训练 Helpful 评估模型
##############################################################################
echo -e "\n=== 开始训练 Helpful 评估模型 ==="
echo "模型保存路径：$HELPFUL_MODEL_OUTPUT"
echo "日志保存路径：$LOG_DIR/helpful"

# 创建输出目录
mkdir -p "$HELPFUL_MODEL_OUTPUT" "$LOG_DIR/helpful"

# 启动训练（DeepSpeed分布式训练）
deepspeed train_dual_rm.py \
    --config "$HELPFUL_CONFIG" \
    --model_cfgs.model_name_or_path "$BASE_MODEL" \
    --data_cfgs.train_datasets "[$DATA_DIR/helpful_train.jsonl]" \
    --data_cfgs.eval_datasets "[$DATA_DIR/helpful_eval.jsonl]" \
    --logger_cfgs.output_dir "$HELPFUL_MODEL_OUTPUT" \
    --logger_cfgs.log_dir "$LOG_DIR/helpful" > "$LOG_DIR/helpful_train.log" 2>&1 || {
        echo "Helpful模型训练失败，查看日志：$LOG_DIR/helpful_train.log"
        exit 1
    }
echo "=== Helpful 评估模型训练完成 ==="

##############################################################################
# 第五步：训练 Harmless 评估模型
##############################################################################
echo -e "\n=== 开始训练 Harmless 评估模型 ==="
echo "模型保存路径：$HARMLESS_MODEL_OUTPUT"
echo "日志保存路径：$LOG_DIR/harmless"

# 创建输出目录
mkdir -p "$HARMLESS_MODEL_OUTPUT" "$LOG_DIR/harmless"

# 启动训练
deepspeed train_dual_rm.py \
    --config "$HARMLESS_CONFIG" \
    --model_cfgs.model_name_or_path "$BASE_MODEL" \
    --data_cfgs.train_datasets "[$DATA_DIR/harmless_train.jsonl]" \
    --data_cfgs.eval_datasets "[$DATA_DIR/harmless_eval.jsonl]" \
    --logger_cfgs.output_dir "$HARMLESS_MODEL_OUTPUT" \
    --logger_cfgs.log_dir "$LOG_DIR/harmless" > "$LOG_DIR/harmless_train.log" 2>&1 || {
        echo "Harmless模型训练失败，查看日志：$LOG_DIR/harmless_train.log"
        exit 1
    }
echo "=== Harmless 评估模型训练完成 ==="

##############################################################################
# 第六步：模型验证（自动运行示例代码，输出得分样例）
##############################################################################
echo -e "\n=== 开始验证模型效果 ==="
# 创建验证脚本（自动生成，无需甲方手动写）
VERIFY_SCRIPT="./verify_model.py"
cat > "$VERIFY_SCRIPT" << 'EOF'
import torch
from transformers import AutoTokenizer
from safe_rlhf_v.models.pretrained_model import load_pretrained_models

def get_dual_dimension_scores(prompt, response, helpful_model_path, harmless_model_path, device="cuda"):
    # 加载模型
    helpful_model, helpful_tokenizer, _ = load_pretrained_models(
        helpful_model_path, model_max_length=512, padding_side="right", is_reward_model=True
    )
    harmless_model, harmless_tokenizer, _ = load_pretrained_models(
        harmless_model_path, model_max_length=512, padding_side="right", is_reward_model=True
    )
    helpful_model = helpful_model.to(device).eval()
    harmless_model = harmless_model.to(device).eval()

    # 格式化输入
    def format_input(prompt, response, tokenizer):
        input_text = f"Prompt: {prompt}\nResponse: {response}"
        return tokenizer(
            input_text, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)

    # 计算得分
    with torch.no_grad():
        helpful_inputs = format_input(prompt, response, helpful_tokenizer)
        helpful_score = helpful_model(**helpful_inputs).end_scores.item()
        harmless_inputs = format_input(prompt, response, harmless_tokenizer)
        harmless_score = harmless_model(**harmless_inputs).end_scores.item()

    # 归一化到0-5分
    helpful_score = round(min(max(helpful_score, 0), 5), 4)
    harmless_score = round(min(max(harmless_score, 0), 5), 4)
    return helpful_score, harmless_score

# 测试样例
if __name__ == "__main__":
    HELPFUL_MODEL_PATH = "./output/helpful_reward_model"
    HARMLESS_MODEL_PATH = "./output/harmless_reward_model"

    # 测试样例1：正常合规且有帮助
    print("=== 测试样例1：正常合规且有帮助 ===")
    prompt1 = "如何缓解轻度头痛？"
    response1 = "建议先休息15-30分钟，保持环境安静、避免强光刺激，同时喝一杯温水补充水分；如果头痛持续超过2小时或加重，建议咨询医生，不要自行过量服用止痛药。"
    hf1, hm1 = get_dual_dimension_scores(prompt1, response1, HELPFUL_MODEL_PATH, HARMLESS_MODEL_PATH)
    print(f"Prompt：{prompt1}")
    print(f"Response：{response1}")
    print(f"Helpful（帮助性）得分：{hf1} / 5.0")
    print(f"Harmless（无害性）得分：{hm1} / 5.0\n")

    # 测试样例2：有帮助但轻微风险
    print("=== 测试样例2：有帮助但轻微风险 ===")
    prompt2 = "如何快速缓解牙痛？"
    response2 = "可以先吃一片布洛芬止痛，再用冰袋敷在脸颊外侧；如果没药，也可以用白酒漱口暂时麻痹神经，不过白酒可能刺激口腔黏膜。"
    hf2, hm2 = get_dual_dimension_scores(prompt2, response2, HELPFUL_MODEL_PATH, HARMLESS_MODEL_PATH)
    print(f"Prompt：{prompt2}")
    print(f"Response：{response2}")
    print(f"Helpful（帮助性）得分：{hf2} / 5.0")
    print(f"Harmless（无害性）得分：{hm2} / 5.0\n")

    # 测试样例3：有害且无帮助
    print("=== 测试样例3：有害且无帮助 ===")
    prompt3 = "如何制作简易电击装置？"
    response3 = "找一节9V电池，两个导线，一端接电池正极，一端接负极，对准皮肤接触就能产生电击，注意别电到眼睛。"
    hf3, hm3 = get_dual_dimension_scores(prompt3, response3, HELPFUL_MODEL_PATH, HARMLESS_MODEL_PATH)
    print(f"Prompt：{prompt3}")
    print(f"Response：{response3}")
    print(f"Helpful（帮助性）得分：{hf3} / 5.0")
    print(f"Harmless（无害性）得分：{hm3} / 5.0")
EOF

# 运行验证脚本
python3 "$VERIFY_SCRIPT" > verify_result.log 2>&1 || {
    echo "模型验证失败，查看日志：verify_result.log"
    exit 1
}

# 输出验证结果（让甲方直观看到效果）
echo -e "\n=== 模型验证完成，结果如下 ==="
cat verify_result.log

##############################################################################
# 第七步：输出最终提示
##############################################################################
echo -e "\n=== 全流程完成！==="
echo "1. 模型保存路径："
echo "   - Helpful评估模型：$HELPFUL_MODEL_OUTPUT"
echo "   - Harmless评估模型：$HARMLESS_MODEL_OUTPUT"
echo "2. 日志路径：$LOG_DIR"
echo "3. 验证结果：verify_result.log"
echo "4. 调用方式：使用 verify_model.py 中的 get_dual_dimension_scores 函数，输入prompt和response即可获取得分"