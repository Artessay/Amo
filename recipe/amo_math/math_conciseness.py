import os
import nltk

MIN_RESPONSE_REWARD_LENGTH = int(os.getenv("MIN_RESPONSE_REWARD_LENGTH", "512"))
MAX_RESPONSE_REWARD_LENGTH = int(os.getenv("MAX_RESPONSE_REWARD_LENGTH", "1024"))

def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text."""
    tokens = nltk.word_tokenize(text)
    return len(tokens)

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Compute the reward score based on the conciseness of the solution."""
    num_tokens = count_tokens(solution_str)

    # linear decay from 1.0 to 0.0 as tokens go from MIN_RESPONSE_REWARD_LENGTH to MAX_RESPONSE_REWARD_LENGTH
    if num_tokens < MIN_RESPONSE_REWARD_LENGTH:
        return 1.0
    elif num_tokens > MAX_RESPONSE_REWARD_LENGTH:
        return 0.0
    else:
        return 1.0 - (num_tokens - MIN_RESPONSE_REWARD_LENGTH) / (MAX_RESPONSE_REWARD_LENGTH - MIN_RESPONSE_REWARD_LENGTH)


if __name__ == "__main__":
    # Test the function with a sample solution
    sample_solution = "The answer is 42."
    score = compute_score("math", sample_solution, "42")
    print(f"Score for '{sample_solution}': {score}")