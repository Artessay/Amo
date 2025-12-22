import argparse
import json
import os
from typing import List, Optional

import torch
from tqdm import tqdm

from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.configs.format_model import ModelFormatter

from build_calibration import (
    compute_end_score,
    extract_prompt,
    extract_responses,
    load_json_dataset,
)


def collect_scores(
    model_path: str,
    dataset_path: str,
    dataset_name: str,
    split: str,
    max_batches: Optional[int],
    output_scores_path: str,
) -> None:
    """Collect raw reward scores on a dataset split and save them to JSON.

    The resulting JSON file has the following minimal structure::

        {"scores": [float, ...], "num_items": int}
    """

    # Load reward model and tokenizer.
    model, tokenizer, _ = load_pretrained_models(
        model_path,
        model_max_length=512,
        padding_side="right",
        is_reward_model=True,
        auto_device_mapping=True,
    )
    model = model.eval()
    formatter = ModelFormatter(tokenizer)

    # Resolve dataset directory and load JSON split.
    base_dir = os.path.join(dataset_path, dataset_name)
    dataset = load_json_dataset(base_dir, split)

    all_scores: List[float] = []
    num_items = 0

    items = dataset
    if max_batches is not None and max_batches > 0:
        items = dataset[:max_batches]

    for item in tqdm(items, desc="Collecting rewards"):
        prompt = extract_prompt(item)
        responses = extract_responses(item)
        if prompt is None or responses is None:
            # Skip items that do not contain the necessary fields.
            continue

        resp0, resp1 = responses
        score0 = compute_end_score(model, tokenizer, formatter, prompt, resp0)
        score1 = compute_end_score(model, tokenizer, formatter, prompt, resp1)

        all_scores.append(float(score0))
        all_scores.append(float(score1))
        num_items += 1

    if not all_scores:
        raise RuntimeError("No valid reward scores were collected from the dataset.")

    payload = {
        "scores": [float(s) for s in all_scores],
        "num_items": int(num_items),
    }

    directory = os.path.dirname(output_scores_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(output_scores_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)

    print(f"Collected {len(all_scores)} scores from {num_items} items.")
    print(f"Scores saved to: {output_scores_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect raw reward scores on a dataset and save them to JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained reward model.",
    )
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
        "--output_scores_path",
        type=str,
        default="rm_raw_scores.json",
        help="Path to the output JSON file for raw reward scores.",
    )

    args = parser.parse_args()

    collect_scores(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        max_batches=args.max_batches,
        output_scores_path=args.output_scores_path,
    )


if __name__ == "__main__":
    main()
