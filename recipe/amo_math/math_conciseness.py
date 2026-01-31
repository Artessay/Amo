import os
import nltk
import numpy as np

from verl.utils.reward_score.math_reward import last_boxed_only_string


EXPECTED_RESPONSE_REWARD_LENGTH = int(os.getenv("EXPECTED_RESPONSE_REWARD_LENGTH", "384"))

def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text."""
    tokens = nltk.word_tokenize(text)
    return len(tokens)

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Compute the reward score based on the conciseness of the solution."""
    num_tokens = count_tokens(solution_str)

    # check if the solution meet the boxed format
    if not last_boxed_only_string(solution_str):
        return 0.0
    
    # exponential decay 
    return np.exp(-num_tokens / EXPECTED_RESPONSE_REWARD_LENGTH)


if __name__ == "__main__":
    # Test the function with a sample solution
    sample_solution = "The answer is 42."
    score = compute_score("math", sample_solution, "42")
    print(f"Score for '{sample_solution}': {score}")