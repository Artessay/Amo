"""Conciseness reward for MATH — redesigned to remove saturation and to create a
genuine trade-off with accuracy.

Problems with the original absolute-threshold version:
* Thresholds were MIN=512 / MAX=1024 tokens, but ~89% of real responses are
  < 512 tokens (median ~314), so conciseness was constant 1.0 — a dead signal.
* It rewarded *any* short answer, including short WRONG ones ("the answer is 5").
  Being concise while wrong is not a virtue, so this created no real tension
  with accuracy.

Redesign:
* **Only score conciseness when the answer is correct.** A wrong answer gets 0.
  This makes accuracy and conciseness genuinely conflicting: to be *correct* on
  a hard problem you usually need a longer chain of reasoning, but to be
  *concise* you must keep it short. That real tension is exactly what a
  multi-objective method (HVPO) is meant to exploit.
* **Thresholds moved onto the real length distribution** (MIN=150, MAX=500,
  roughly p25..p90 of observed correct answers) so the reward actually varies.
* **Smooth (no flat plateaus):** a smooth cosine ramp from 1.0 (<=MIN) to 0.0
  (>=MAX) instead of a hard piecewise-linear step, giving a continuous gradient
  everywhere in the active band.

All thresholds are env-overridable for ablation.
"""

import math
import os

MIN_RESPONSE_REWARD_LENGTH = int(os.getenv("MIN_RESPONSE_REWARD_LENGTH", "150"))
MAX_RESPONSE_REWARD_LENGTH = int(os.getenv("MAX_RESPONSE_REWARD_LENGTH", "500"))
# When True, conciseness is only credited to correct answers (recommended).
CONCISENESS_REQUIRE_CORRECT = os.getenv("CONCISENESS_REQUIRE_CORRECT", "1") == "1"


def count_tokens(text: str) -> int:
    """Count tokens; prefer nltk, fall back to whitespace split if unavailable."""
    try:
        import nltk

        return len(nltk.word_tokenize(text))
    except Exception:
        return len(text.split())


def _length_score(num_tokens: int) -> float:
    """Smooth conciseness in [0, 1]: 1.0 for short, 0.0 for long, cosine ramp."""
    lo, hi = MIN_RESPONSE_REWARD_LENGTH, MAX_RESPONSE_REWARD_LENGTH
    if num_tokens <= lo:
        return 1.0
    if num_tokens >= hi:
        return 0.0
    # Cosine ease-out from 1 -> 0 over [lo, hi] (smooth, no flat plateau inside).
    frac = (num_tokens - lo) / (hi - lo)
    return 0.5 * (1.0 + math.cos(math.pi * frac))


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Conciseness reward. 0 for wrong answers (when REQUIRE_CORRECT); else a
    smooth function of length that decreases as the answer gets longer."""
    if CONCISENESS_REQUIRE_CORRECT:
        # Lazy import to avoid a hard dependency cycle at module load.
        from verl.utils.reward_score.math_reward import compute_score as _acc
        try:
            correct = float(_acc(solution_str, ground_truth)) >= 1.0
        except Exception:
            correct = False
        if not correct:
            return 0.0

    return _length_score(count_tokens(solution_str))


if __name__ == "__main__":
    # correct answers of varying length
    short_correct = "Reasoning... therefore \\boxed{42}."
    long_correct = ("Step one we do this. " * 60) + " \\boxed{42}."
    wrong = "The answer is \\boxed{7}."
    for name, sol, gt in [
        ("short+correct", short_correct, "42"),
        ("long+correct", long_correct, "42"),
        ("short+wrong", wrong, "42"),
    ]:
        print(f"{name:14} tokens={count_tokens(sol):4d} score={compute_score('math', sol, gt):.3f}")
