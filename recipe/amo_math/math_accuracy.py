from verl.utils.reward_score.math_reward import compute_score as compute_accuracy_boxed

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer

    Returns:
        Reward score (1.0 for correct, 0.0 for incorrect)
    """
    return compute_accuracy_boxed(solution_str, ground_truth)