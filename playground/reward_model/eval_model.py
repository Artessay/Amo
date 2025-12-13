import os
import torch
import json
from tqdm import tqdm
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.configs.format_model import ModelFormatter
from align_anything.utils.tools import seed_everything

class RewardEvaluator:
    """
    RewardEvaluator holds the reward model and tokenizer in memory
    to avoid reloading the model for every evaluation.
    """
    def __init__(self, model_path):
        """
        Initialize the reward model and tokenizer. Load only once.
        """
        self.model_path = model_path
        self.model, self.tokenizer, _ = load_pretrained_models(
            model_path,
            model_max_length=512,
            padding_side="right",
            is_reward_model=True,
            auto_device_mapping=True,
        )
        self.model = self.model.eval()
        self.formatter = ModelFormatter(self.tokenizer)

    def get_reward_score(self, prompt, response):
        """
        Compute reward score for a prompt-response pair.
        """
        conversation = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response},
        ]
        formatted = self.formatter(conversation)

        inputs = self.tokenizer(
            formatted,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model(**inputs)
            score = output.end_scores.item()
        
        return score

    def evaluate(
        self,
        label_key, 
        dataset_name="PKU-SafeRLHF",
        save_results=True,
        results_dir="results"
    ):
        """
        Evaluate the reward model accuracy for a specified label key.
        label_key specifies which field to use as the ground-truth label.
        After evaluation, save the results as a json file in results_dir.
        Returns: result_dict (dict)
        """
        with open(f"./{dataset_name}/test.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        correct = 0
        total = 0
        for item in tqdm(dataset, desc="Evaluating"):
            prompt = item["prompt"]
            response_0 = item["response_0"]
            response_1 = item["response_1"]
            label = item[label_key]
            score_0 = self.get_reward_score(prompt, response_0)
            score_1 = self.get_reward_score(prompt, response_1)
            # Select the response with a higher score
            pred = 0 if score_0 > score_1 else 1
            if pred == label:
                correct += 1
            total += 1
        accuracy = correct / total if total > 0 else 0.0

        result_dict = {
            "model_path": self.model_path,
            "label_key": label_key,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

        # Save results to file
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            result_file = os.path.join(
                results_dir,
                f"reward_{dataset_name}_{os.path.basename(self.model_path)}_{label_key}.json"
            )
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4)
            print(f"Results saved to {result_file}")

        print(result_dict)
        return result_dict


if __name__ == "__main__":
    seed_everything(42)

    dataset_name = "PKU-SafeRLHF"
    helpful_model_path = "./checkpoints/Qwen3-1.7B-SafeRLHF-RM"
    harmless_model_path = "./checkpoints/Qwen3-1.7B-SafeRLHF-CM"
    
    # Evaluate helpfulness
    helpful_evaluator = RewardEvaluator(helpful_model_path)
    helpful_evaluator.evaluate(
        label_key="better_response_id",
        dataset_name="PKU-SafeRLHF",
        results_dir="results"
    )

    # Evaluate harmlessness
    harmless_evaluator = RewardEvaluator(harmless_model_path)
    harmless_evaluator.evaluate(
        label_key="safer_response_id",
        dataset_name="PKU-SafeRLHF",
        results_dir="results"
    )