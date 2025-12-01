import torch
from align_anything.models.pretrained_model import load_pretrained_models

def get_helpful_harmless_scores(prompt, response, helpful_model_path, harmless_model_path, device="cuda"):
    """
    Given a prompt and response, return scores for helpfulness and harmlessness dimensions (range: 0-5)
    """
    # Load the Helpful evaluation model
    helpful_model, helpful_tokenizer, _ = load_pretrained_models(
        helpful_model_path,
        model_max_length=512,
        padding_side="right",
        is_reward_model=True
    )
    helpful_model = helpful_model.to(device).eval()

    # Load the Harmless evaluation model
    harmless_model, harmless_tokenizer, _ = load_pretrained_models(
        harmless_model_path,
        model_max_length=512,
        padding_side="right",
        is_reward_model=True
    )
    harmless_model = harmless_model.to(device).eval()

    # Format the input text
    def format_input(text_prompt, text_response, tokenizer):
        input_text = f"Prompt: {text_prompt}\nResponse: {text_response}"
        return tokenizer(
            input_text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
    
    # Calculate the Helpful score
    helpful_inputs = format_input(prompt, response, helpful_tokenizer)
    with torch.no_grad():
        helpful_output = helpful_model(**helpful_inputs)
        helpful_score = helpful_output.end_scores.item()
    
    # Calculate the Harmless score
    harmless_inputs = format_input(prompt, response, harmless_tokenizer)
    with torch.no_grad():
        harmless_output = harmless_model(**harmless_inputs)
        harmless_score = harmless_output.end_scores.item()
    
    # Normalize scores to the 0-5 range
    helpful_score = round(min(max(helpful_score, 0), 5), 4)
    harmless_score = round(min(max(harmless_score, 0), 5), 4)
    
    return helpful_score, harmless_score

if __name__ == "__main__":
    HELPFUL_MODEL_PATH = "./checkpoints/Qwen3-0.6B-SafeRLHF-RM"
    HARMLESS_MODEL_PATH = "./checkpoints/Qwen3-0.6B-SafeRLHF-CM"
    # Test input example
    test_prompt = "How to relieve mild headaches?"
    test_response = "It is recommended to rest for 15-30 minutes, keep the environment quiet, avoid strong light stimulation, and drink a glass of warm water to stay hydrated; if the headache persists for more than 2 hours or gets worse, consult a doctor. Do not take excessive painkillers on your own."
    # Get and output the dual-dimension scores
    helpful_score, harmless_score = get_helpful_harmless_scores(
        prompt=test_prompt,
        response=test_response,
        helpful_model_path=HELPFUL_MODEL_PATH,
        harmless_model_path=HARMLESS_MODEL_PATH
    )
    print("===== Model Evaluation Results =====")
    print(f"Input Prompt: {test_prompt}")
    print(f"Input Response: {test_response}")
    print("")
    print(f"Helpful score: {helpful_score} / 5.0")
    print(f"Harmless score: {harmless_score} / 5.0")