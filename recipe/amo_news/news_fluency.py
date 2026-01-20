"""UniEval-based fluency metric for CNN/DailyMail summarization.

This module exposes a `compute_score` function with the standard Amo
signature used by the offline evaluator and reward manager.
"""

from recipe.amo_news.summarization_client import evaluate_summarization

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """Compute UniEval fluency score for a single summary.

    Args:
        data_source: Name of the dataset (e.g. "cnn_dailymail"). Unused but
            kept for a uniform Amo metric signature.
        solution_str: Model-generated summary to be evaluated.
        ground_truth: Reference summary (highlights). Unused for fluency.
        extra_info: Per-example metadata. For CNN/DailyMail this must include
            the original `article` string.

    Returns:
        A float fluency score in [0, 1] produced by UniEval.
    """
    del data_source  # Unused but kept for a uniform Amo metric signature.

    return evaluate_summarization(solution_str, ground_truth, extra_info, dim="fluency")
