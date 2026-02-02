import os
import re
import numpy as np

# from verl.utils.reward_score.math_reward import last_boxed_only_string
from verl.utils.reward_score.math_reward import compute_score as compute_accuracy_boxed

# Boxed answer: at least one non-blank char inside \boxed{…}
BOXED_RE = re.compile(r"\\boxed\{\s*[^}\s][^}]*\}", re.S)

EXPECTED_RESPONSE_REWARD_LENGTH = int(os.getenv("EXPECTED_RESPONSE_REWARD_LENGTH", "384"))


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Compute the reward score based on the conciseness of the solution."""
    num_tokens = extra_info.get("num_tokens", 0)
    assert num_tokens > 0, "num_tokens should be greater than 0"

    # check if the solution meet the requirement
    if not BOXED_RE.search(solution_str):
        return 0.0
    
    # exponential decay 
    return np.exp(-num_tokens / EXPECTED_RESPONSE_REWARD_LENGTH)


if __name__ == "__main__":
    # Test the function with a sample solution
    sample_solution = "The answer is \\boxed{42}."
    extra_info = {"num_tokens": 10}
    score = compute_score("math", sample_solution, "42", extra_info)
    print(f"Score for '{sample_solution}': {score}")