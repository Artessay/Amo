import torch
from align_anything.models.pretrained_model import load_pretrained_models

MAX_PROMPT_LENGTH = 1024
REWARD_MODEL_PATH = '/data/PKU-Alignment/beaver-7b-v3.0-reward'

helpful_model, helpful_tokenizer, _ = load_pretrained_models(
    REWARD_MODEL_PATH,
    model_max_length=MAX_PROMPT_LENGTH,
    padding_side="right",
    auto_device_mapping=True,
    is_reward_model=True,
)


def compute_helpful_score(prompt, response) -> float:
    input = f"Prompt: {prompt}\nResponse: {response}"
    # input = 'BEGINNING OF CONVERSATION: USER: hello ASSISTANT:Hello! How can I help you today?'

    helpful_inputs = helpful_tokenizer(input, return_tensors='pt')
    with torch.no_grad():
        helpful_output = helpful_model(**helpful_inputs)
        helpful_score = helpful_output.end_scores.item()

    print(helpful_score)