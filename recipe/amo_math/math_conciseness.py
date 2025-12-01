import nltk

MAX_RESPONSE_LENGTH = 4096

def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text."""
    tokens = nltk.word_tokenize(text)
    return len(tokens)

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Compute the reward score based on the conciseness of the solution."""
    num_tokens = count_tokens(solution_str)

    # linear decay from 1.0 to 0.0 as tokens go from 0 to MAX_RESPONSE_LENGTH
    return max(0.0, 1.0 - (num_tokens / MAX_RESPONSE_LENGTH))

