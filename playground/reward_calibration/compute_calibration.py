import argparse
import json
import os

import torch

from align_anything.utils.reward_calibration import estimate_alpha_beta, save_calibration


def load_scores(scores_path: str) -> torch.Tensor:
    """Load raw reward scores from a JSON file.

    The JSON file is expected to contain a dictionary of the form::

        {"scores": [float, ...], "num_items": int}
    """

    if not os.path.isfile(scores_path):
        raise FileNotFoundError(f"Scores file not found: {scores_path}")

    with open(scores_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Scores file must contain a JSON object.")

    if "scores" not in data or "num_items" not in data:
        raise KeyError("Scores JSON must contain 'scores' and 'num_items' keys.")

    scores = data["scores"]
    if not isinstance(scores, list):
        raise TypeError("'scores' field in JSON must be a list.")

    if len(scores) == 0:
        raise ValueError("Scores list is empty.")

    scores_tensor = torch.tensor([float(s) for s in scores], dtype=torch.float32)
    return scores_tensor


def compute_calibration(scores_path: str, p: float, output_path: str) -> None:
    """Compute calibration parameters (alpha, beta) from saved scores."""

    if not (0.0 < p < 0.5):
        raise ValueError("Argument p must be in the open interval (0, 0.5).")

    scores_tensor = load_scores(scores_path)
    alpha, beta = estimate_alpha_beta(scores_tensor, p=p)

    print(f"Loaded {scores_tensor.numel()} scores from: {scores_path}")
    print(f"Estimated calibration parameters: alpha={alpha:.6f}, beta={beta:.6f}")

    save_calibration(output_path, alpha=alpha, beta=beta)
    print(f"Calibration saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute reward calibration parameters from saved raw scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scores_path",
        type=str,
        required=True,
        help="Path to the JSON file containing raw reward scores.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.1,
        help="Lower-tail probability used for quantile constraints.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="rm_calibration.json",
        help="Path to the output JSON calibration file.",
    )

    args = parser.parse_args()

    compute_calibration(
        scores_path=args.scores_path,
        p=args.p,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
