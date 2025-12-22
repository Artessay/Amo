import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.configs.format_model import ModelFormatter
from align_anything.utils.reward_calibration import estimate_alpha_beta, save_calibration


def load_json_dataset(base_dir: str, split: str) -> List[Dict[str, Any]]:
    """Load a simple JSON dataset file ``<split>.json`` from ``base_dir``.

    The JSON file is expected to contain a list of items (dictionaries).
    """

    json_path = os.path.join(base_dir, f"{split}.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Dataset file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of items in {json_path}, got {type(data)!r}.")

    return data


def extract_prompt(item: Dict[str, Any]) -> Optional[str]:
    """Extract a plain-text prompt from a dataset item.

    This helper is intentionally permissive:

    * If ``item["prompt"]`` is a string, it is returned directly.
    * If it is a list of messages (each containing a ``"content"`` field),
      their contents are concatenated with blank lines.
    """

    prompt = item.get("prompt")
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        contents: List[str] = []
        for message in prompt:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    contents.append(content)
        if contents:
            return "\n\n".join(contents)

    return None


def extract_responses(item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Extract a pair of responses from a dataset item.

    The function tries a few common field patterns and returns the first
    successful pair it finds. If it cannot find two responses, it returns
    ``None`` and the caller is expected to skip this item.
    """

    # PKU-SafeRLHF style.
    r0 = item.get("response_0")
    r1 = item.get("response_1")
    if isinstance(r0, str) and isinstance(r1, str):
        return r0, r1

    # Generic "chosen" / "rejected" style.
    chosen = item.get("chosen")
    rejected = item.get("rejected")
    if isinstance(chosen, str) and isinstance(rejected, str):
        return chosen, rejected

    # Fallback: some datasets may use "chosen_response" / "rejected_response".
    chosen = item.get("chosen_response")
    rejected = item.get("rejected_response")
    if isinstance(chosen, str) and isinstance(rejected, str):
        return chosen, rejected

    return None


def compute_end_score(
    model, tokenizer, formatter: ModelFormatter, prompt: str, response: str
) -> float:
    """Compute the raw end_score for a single prompt/response pair."""

    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    formatted = formatter(conversation)

    inputs = tokenizer(
        formatted,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model(**inputs)
        # end_scores is expected to have shape (B, 1) or (B,) for B=1.
        score_tensor = output.end_scores
        score = score_tensor.view(-1)[0].item()

    return float(score)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a shift & temperature calibration for a reward model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained reward model.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Root path to the dataset directory.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PKU-SafeRLHF",
        help="Name of the dataset subdirectory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (e.g., 'train' or 'test').",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional limit on the number of dataset items to process.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="rm_calibration.json",
        help="Path to the output JSON calibration file.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.1,
        help="Lower-tail probability used for quantile constraints.",
    )

    args = parser.parse_args()

    if not (0.0 < args.p < 0.5):
        raise ValueError("Argument --p must be in the open interval (0, 0.5).")

    # Load reward model and tokenizer.
    model, tokenizer, _ = load_pretrained_models(
        args.model_path,
        model_max_length=512,
        padding_side="right",
        is_reward_model=True,
        auto_device_mapping=True,
    )
    model = model.eval()
    formatter = ModelFormatter(tokenizer)

    # Resolve dataset directory and load JSON split.
    base_dir = os.path.join(args.dataset_path, args.dataset_name)
    dataset = load_json_dataset(base_dir, args.split)

    all_scores: List[float] = []
    num_items = 0

    iterator = enumerate(dataset)
    if args.max_batches is not None and args.max_batches > 0:
        iterator = enumerate(dataset[: args.max_batches])

    for idx, item in tqdm(list(iterator), desc="Collecting rewards"):
        prompt = extract_prompt(item)
        responses = extract_responses(item)
        if prompt is None or responses is None:
            # Skip items that do not contain the necessary fields.
            continue

        resp0, resp1 = responses
        score0 = compute_end_score(model, tokenizer, formatter, prompt, resp0)
        score1 = compute_end_score(model, tokenizer, formatter, prompt, resp1)

        all_scores.append(score0)
        all_scores.append(score1)
        num_items += 1

    if not all_scores:
        raise RuntimeError("No valid reward scores were collected from the dataset.")

    r_all = torch.tensor(all_scores, dtype=torch.float32)
    alpha, beta = estimate_alpha_beta(r_all, p=args.p)

    print(f"Collected {len(all_scores)} scores from {num_items} items.")
    print(f"Estimated calibration parameters: alpha={alpha:.6f}, beta={beta:.6f}")

    save_calibration(args.output_path, alpha=alpha, beta=beta)
    print(f"Calibration saved to: {args.output_path}")


if __name__ == "__main__":
    main()
