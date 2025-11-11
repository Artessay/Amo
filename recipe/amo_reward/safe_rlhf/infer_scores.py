import torch
from pretrained_model import load_pretrained_models

def get_helpful_harmless_scores(prompt, response, helpful_model_path, harmless_model_path, device="cuda"):
    """
    输入prompt和response，返回Helpful和Harmless维度的得分（0-5分）
    """
    # 加载Helpful评估模型
    helpful_model, helpful_tokenizer, _ = load_pretrained_models(
        helpful_model_path,
        model_max_length=512,
        padding_side="right",
        is_reward_model=True
    )
    helpful_model = helpful_model.to(device).eval()

    # 加载Harmless评估模型
    harmless_model, harmless_tokenizer, _ = load_pretrained_models(
        harmless_model_path,
        model_max_length=512,
        padding_side="right",
        is_reward_model=True
    )
    harmless_model = harmless_model.to(device).eval()

    # 格式化输入文本
    def format_input(text_prompt, text_response, tokenizer):
        input_text = f"Prompt: {text_prompt}\nResponse: {text_response}"
        return tokenizer(
            input_text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

    # 计算Helpful得分
    helpful_inputs = format_input(prompt, response, helpful_tokenizer)
    with torch.no_grad():
        helpful_output = helpful_model(**helpful_inputs)
        helpful_score = helpful_output.end_scores.item()

    # 计算Harmless得分
    harmless_inputs = format_input(prompt, response, harmless_tokenizer)
    with torch.no_grad():
        harmless_output = harmless_model(**harmless_inputs)
        harmless_score = harmless_output.end_scores.item()

    # 归一化到0-5分区间
    helpful_score = round(min(max(helpful_score, 0), 5), 4)
    harmless_score = round(min(max(harmless_score, 0), 5), 4)

    return helpful_score, harmless_score

if __name__ == "__main__":
    # 训练好的模型路径（需与训练脚本的输出路径一致）
    HELPFUL_MODEL_PATH = "./output/helpful_reward_model"
    HARMLESS_MODEL_PATH = "./output/harmless_reward_model"

    # 测试输入示例
    test_prompt = "如何缓解轻度头痛？"
    test_response = "建议先休息15-30分钟，保持环境安静、避免强光刺激，同时喝一杯温水补充水分；如果头痛持续超过2小时或加重，建议咨询医生，不要自行过量服用止痛药。"

    # 获取并输出双维度得分
    helpful_score, harmless_score = get_helpful_harmless_scores(
        prompt=test_prompt,
        response=test_response,
        helpful_model_path=HELPFUL_MODEL_PATH,
        harmless_model_path=HARMLESS_MODEL_PATH
    )

    print("===== 模型得分评估结果 =====")
    print(f"输入Prompt：{test_prompt}")
    print(f"输入Response：{test_response}")
    print("")
    print(f"Helpful（帮助性）得分：{helpful_score} / 5.0")
    print(f"Harmless（无害性）得分：{harmless_score} / 5.0")