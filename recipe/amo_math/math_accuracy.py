"""Dense accuracy reward for MATH.

The original reward was a hard 0/1 (correct/incorrect). On a rollout group the
samples are frequently *all* correct or *all* wrong, which collapses the
group-relative advantage (mean-centred reward becomes ~0) and starves both GRPO
and HVPO of a learning signal.

This dense version decomposes the correctness signal into orthogonal,
partially-credited components so the group almost always carries some variance:

    accuracy = 0.1 * [produced a \\boxed{...} answer]          (format / attempt)
             + 0.9 * [answer is equivalent to ground truth]     (true correctness)
             + eps * [numeric closeness]  (only when WRONG and both parse to numbers)

Design notes:
* The "true correctness" term reuses the original, well-tested ``compute_score``
  (handles fractions, sqrt, latex, string normalization), so a correct answer
  always scores >= 1.0 (0.1 format + 0.9 correct).
* The numeric-closeness term (``eps=0.1``) only fires when the answer is WRONG
  and both the prediction and the ground truth parse to finite numbers. It gives
  a small gradient toward the right magnitude without ever out-weighing an
  actually-correct answer. It is deliberately tiny to avoid rewarding "guess a
  nearby number" over genuine solving.
* Score is in [0, 1]: wrong+no-box = 0.0, wrong+box = 0.1(+<=0.1 numeric),
  correct = 1.0.
"""

import os
import re

from verl.utils.reward_score.math_reward import (
    compute_score as compute_accuracy_boxed,
    last_boxed_only_string,
    remove_boxed,
)

FORMAT_WEIGHT = float(os.getenv("MATH_ACC_FORMAT_WEIGHT", "0.1"))
CORRECT_WEIGHT = float(os.getenv("MATH_ACC_CORRECT_WEIGHT", "0.9"))
NUMERIC_WEIGHT = float(os.getenv("MATH_ACC_NUMERIC_WEIGHT", "0.1"))


def _parse_number(s: str):
    """Best-effort parse of an answer string into a finite float; else None."""
    if s is None:
        return None
    t = s.strip().replace(",", "").replace("\\!", "").replace(" ", "")
    # Strip common latex wrappers that do not change the numeric value.
    t = re.sub(r"\\(text|mathrm|left|right|,)", "", t)
    t = t.replace("\\%", "").replace("%", "")
    try:
        v = float(t)
    except (ValueError, TypeError):
        try:
            from sympy import sympify

            v = float(sympify(t))
        except Exception:
            return None
    # Reject non-finite values.
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def _has_boxed(solution_str: str) -> bool:
    try:
        return last_boxed_only_string(solution_str) is not None
    except Exception:
        return False


def _boxed_answer(solution_str: str):
    try:
        box = last_boxed_only_string(solution_str)
        return remove_boxed(box) if box is not None else None
    except Exception:
        return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Dense MATH accuracy in [0, 1] (see module docstring)."""
    has_box = _has_boxed(solution_str)

    # True correctness (reuses the original robust equivalence check).
    try:
        correct = float(compute_accuracy_boxed(solution_str, ground_truth)) >= 1.0
    except Exception:
        correct = False

    score = FORMAT_WEIGHT * float(has_box) + CORRECT_WEIGHT * float(correct)

    # Numeric closeness only when wrong and both sides parse to finite numbers.
    if not correct and has_box:
        pred = _parse_number(_boxed_answer(solution_str))
        gt = _parse_number(ground_truth)
        if pred is not None and gt is not None and gt != 0.0:
            rel_err = abs(pred - gt) / abs(gt)
            score += NUMERIC_WEIGHT * max(0.0, 1.0 - rel_err)

    return float(min(1.0, max(0.0, score)))


if __name__ == "__main__":
    tests = [
        (r"the answer is \boxed{17}", "17"),           # correct -> 1.0
        (r"the answer is \boxed{17.5}", "17"),          # wrong, close -> ~0.2
        (r"the answer is \boxed{100}", "17"),           # wrong, far  -> 0.1
        (r"no box at all", "17"),                        # no attempt -> 0.0
        (r"the answer is \boxed{\frac{1}{2}}", "\\frac{1}{2}"),  # correct fraction
    ]
    for sol, gt in tests:
        print(f"score={compute_score('math', sol, gt):.3f}  gt={gt!r}  sol={sol[:35]!r}")
