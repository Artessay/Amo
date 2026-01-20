"""Common UniEval helpers for CNN/DailyMail summarization metrics.

This module centralizes UniEval imports, evaluator caching and per-dimension
configuration so that individual metric files can stay as thin wrappers.
"""

from __future__ import annotations

from typing import Any

try:
    # UniEval entry points
    from recipe.amo_news.metric.evaluator import SumEvaluator
    from recipe.amo_news.metric.utils import convert_to_json
except ImportError as e:  # pragma: no cover - optional runtime dependency
    raise ImportError(
        "UniEval is required for CNN/DailyMail summarization metrics. "
        "Please install it via `pip install -r Amo/requirements.txt`."
    ) from e



def extract_dimension_score(result: Any, dim: str) -> float:
    """Extract a single-dimension score from UniEval evaluation output.

    UniEval typically returns either:
      * a list of per-example dicts, or
      * a dict of dimension -> value or [value].
    """

    # List-of-dicts case
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and dim in first:
            return float(first[dim])

    # Dict case
    if isinstance(result, dict) and dim in result:
        value = result[dim]
        if isinstance(value, list) and value:
            return float(value[0])
        return float(value)

    raise RuntimeError(f"Unexpected UniEval output format when extracting dimension '{dim}'.")


def _require_article(extra_info: Any, dim: str) -> str:
    if not isinstance(extra_info, dict):
        raise ValueError(
            f"extra_info must be a dict containing an 'article' field when "
            f"evaluating '{dim}' with UniEval."
        )

    article = extra_info.get("article")
    if not isinstance(article, str) or not article.strip():
        raise ValueError(
            f"extra_info['article'] must be a non-empty string when evaluating '{dim}' with UniEval."
        )
    return article


def _require_ground_truth(ground_truth: Any, dim: str) -> str:
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError(
            f"ground_truth must be a non-empty string when evaluating '{dim}' with UniEval."
        )
    return ground_truth


def evaluate_dimension(evaluator: SumEvaluator, solution_str: str, ground_truth: str, extra_info: dict | None, dim: str) -> float:
    """Evaluate a single UniEval summarization dimension.

    Args:
        solution_str: Model-generated summary.
        ground_truth: Reference summary (used by 'relevance'; ignored otherwise).
        extra_info: Optional metadata; for 'coherence' and 'consistency' this
            must contain an 'article' field with the source document.
        dim: One of {"coherence", "consistency", "fluency", "relevance"}.
    """

    dim = dim.lower()

    if dim in {"coherence", "consistency"}:
        article = _require_article(extra_info, dim)
        data = convert_to_json(output_list=[solution_str], src_list=[article])
    elif dim == "fluency":
        data = convert_to_json(output_list=[solution_str])
    elif dim == "relevance":
        ref = _require_ground_truth(ground_truth, dim)
        data = convert_to_json(output_list=[solution_str], ref_list=[ref])
    else:
        raise ValueError(f"Unsupported summarization dimension '{dim}'.")

    result = evaluator.evaluate(data, dims=[dim], overall=False, print_result=False)
    return extract_dimension_score(result, dim)

if __name__ == "__main__":
    from recipe.amo_news.metric.evaluator import get_evaluator

    article = "London (CNN)A 19-year-old man was charged Wednesday with terror offenses after he was arrested as he returned to Britain from Turkey, London's Metropolitan Police said. Yahya Rashid, a UK national from northwest London, was detained at Luton airport on Tuesday after he arrived on a flight from Istanbul, police said. He's been charged with engaging in conduct in preparation of acts of terrorism, and with engaging in conduct with the intention of assisting others to commit acts of terrorism. Both charges relate to the period between November 1 and March 31. Rashid is due to appear in Westminster Magistrates' Court on Wednesday, police said. CNN's Lindsay Isaac contributed to this report."
    solution_str = "A 19-year-old UK national, Yahya Rashid, was charged with terror offenses after being arrested at Luton airport upon his return from Turkey, according to London's Metropolitan Police. Rashid faces charges of preparing for acts of terrorism and intending to assist others in committing terrorism between November 1 and March 31. He is scheduled to appear in Westminster Magistrates' Court."
    ground_truth = "London's Metropolitan Police say the man was arrested at Luton airport after landing on a flight from Istanbul .\nHe's been charged with terror offenses allegedly committed since the start of November ."

    evaluator = get_evaluator("summarization")
    extra_info = {"article": article}

    dim = "relevance"

    score = evaluate_dimension(
        evaluator=evaluator,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        dim=dim,
    )
    print(f"Score for dimension '{dim}': {score}")