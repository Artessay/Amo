"""Common UniEval helpers for CNN/DailyMail summarization metrics.

This module centralizes UniEval imports, evaluator caching and per-dimension
configuration so that individual metric files can stay as thin wrappers.
"""

from __future__ import annotations

from typing import Any

try:
    # UniEval entry points
    from recipe.amo_news.metric.evaluator import get_evaluator
    from recipe.amo_news.metric.utils import convert_to_json
except ImportError as e:  # pragma: no cover - optional runtime dependency
    raise ImportError(
        "UniEval is required for CNN/DailyMail summarization metrics. "
        "Please install it via `pip install -r Amo/requirements.txt`."
    ) from e


_EVALUATOR = None


def get_summarization_evaluator():
    """Return a cached UniEval summarization evaluator instance."""
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = get_evaluator("summarization")
    return _EVALUATOR


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


def evaluate_dimension(solution_str: str, ground_truth: str, extra_info: dict | None, dim: str) -> float:
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

    evaluator = get_summarization_evaluator()
    result = evaluator.evaluate(data, dims=[dim], overall=False, print_result=False)
    return extract_dimension_score(result, dim)
