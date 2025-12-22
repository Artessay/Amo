import os
import torch
import json
from typing import Optional, Tuple

from tqdm import tqdm

from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.configs.format_model import ModelFormatter
from align_anything.utils.reward_calibration import calibrate_score, load_calibration
from align_anything.utils.tools import seed_everything


class RewardEvaluator:
    """Evaluate a reward model on prompt-response pairs.

    This helper holds the reward model and tokenizer in memory to avoid
    reloading the model for every evaluation.
    """

    def __init__(self, model_path: str, calibration_path: Optional[str] = None) -> None:
        """Initialize the reward model and tokenizer.

        Args:
            model_path: Path to the trained reward model checkpoint.
            calibration_path: Optional path to a JSON file containing
                calibration parameters {"alpha": float, "beta": float}. If
                provided and the file exists, the evaluator will return
                calibrated scores in [0, 1] instead of raw logits.
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

        self._calibration: Optional[Tuple[float, float]] = None
        if calibration_path is not None:
            if os.path.isfile(calibration_path):
                alpha, beta = load_calibration(calibration_path)
                self._calibration = (alpha, beta)
                print(
                    f"[RewardEvaluator] Loaded calibration from {calibration_path} "
                    f"(alpha={alpha:.6f}, beta={beta:.6f})."
                )
            else:
                print(
                    f"[RewardEvaluator] Calibration file not found at {calibration_path}; "
                    "continuing without calibration."
                )

    def _compute_raw_score(self, prompt: str, response: str) -> float:
        """Compute the raw end_score (logit) for a prompt-response pair."""

        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        formatted = self.formatter(conversation)

        inputs = self.tokenizer(
            formatted,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model(**inputs)
            score_tensor = output.end_scores
            score = score_tensor.view(-1)[0].item()

        return float(score)

    def get_reward_score(self, prompt: str, response: str) -> float:
        """Compute a (raw or calibrated) reward score for a single pair.

        If calibration parameters were provided at initialization time, this
        returns the calibrated score in [0, 1]. Otherwise it returns the raw
        logit from the reward model.
        """

        raw_score = self._compute_raw_score(prompt, response)

        if self._calibration is None:
            return raw_score

        alpha, beta = self._calibration
        raw_tensor = torch.tensor([raw_score], dtype=torch.float32)
        calibrated_tensor = calibrate_score(raw_tensor, alpha=alpha, beta=beta)
        return float(calibrated_tensor.item())

    def get_reward_score_both(self, prompt: str, response: str) -> Tuple[float, float]:
        """Return both raw and calibrated scores for a prompt-response pair.

        When calibration is not active, this returns ``(raw, raw)`` so that
        callers can rely on a fixed two-element tuple.
        """

        raw_score = self._compute_raw_score(prompt, response)

        if self._calibration is None:
            return raw_score, raw_score

        alpha, beta = self._calibration
        raw_tensor = torch.tensor([raw_score], dtype=torch.float32)
        calibrated_tensor = calibrate_score(raw_tensor, alpha=alpha, beta=beta)
        calibrated_score = float(calibrated_tensor.item())
        return raw_score, calibrated_score

    def evaluate(
        self,
        label_key: str,
        dataset_name: str = "PKU-SafeRLHF",
        save_results: bool = True,
        results_dir: str = "results",
    ) -> dict:
        """Evaluate pairwise accuracy on a given dataset.

        Args:
            label_key: Field name that stores the ground-truth preferred
                response index (e.g. "better_response_id" or "safer_response_id").
            dataset_name: Directory name under the current working directory
                that contains ``test.json``.
            save_results: Whether to save the evaluation summary as JSON.
            results_dir: Directory to store the result JSON file.

        Returns:
            A dictionary with evaluation statistics.
        """

        dataset_path = os.path.join(f"./{dataset_name}", "test.json")
        with open(dataset_path, "r", encoding="utf-8") as f:
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

            # Select the response with a higher score.
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
            "enable_calibration": self._calibration is not None,
        }

        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            result_file = os.path.join(
                results_dir,
                f"reward_{dataset_name}_{os.path.basename(self.model_path)}_{label_key}.json",
            )
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4)
            print(f"Results saved to {result_file}")

        if self._calibration is not None:
            alpha, beta = self._calibration
            print(
                f"Evaluation completed with calibration active: "
                f"alpha={alpha:.6f}, beta={beta:.6f}."
            )

        print(result_dict)
        return result_dict


if __name__ == "__main__":
    seed_everything(42)

    dataset_name = "PKU-SafeRLHF"
    helpful_model_path = "./checkpoints/Qwen2.5-7B-SafeRLHF-RM"
    harmless_model_path = "./checkpoints/Qwen2.5-7B-SafeRLHF-CM"

    # If you have precomputed calibration parameters, set the paths here.
    helpful_calibration_path = None  # e.g. "./checkpoints/helpful_rm_calibration.json"
    harmless_calibration_path = None  # e.g. "./checkpoints/harmless_rm_calibration.json"

    # Evaluate helpfulness.
    helpful_evaluator = RewardEvaluator(
        helpful_model_path,
        calibration_path=helpful_calibration_path,
    )
    helpful_evaluator.evaluate(
        label_key="better_response_id",
        dataset_name=dataset_name,
        results_dir="results",
    )

    # Evaluate harmlessness.
    harmless_evaluator = RewardEvaluator(
        harmless_model_path,
        calibration_path=harmless_calibration_path,
    )
    harmless_evaluator.evaluate(
        label_key="safer_response_id",
        dataset_name=dataset_name,
        results_dir="results",
    )
