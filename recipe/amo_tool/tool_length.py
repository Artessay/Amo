# rlla_length_score.py
import os


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """
    Compute RLLA length reward (based on word-count inside <think>...</think>).
    ground_truth is unused here (kept for a uniform interface).
    """
    response = solution_str

    max_possible_reward = 1.0
    min_possible_reward = 0.0
    max_reward_len = 512

    if "<think>" not in response or "</think>" not in response:
        return float(min_possible_reward)

    think_text = response.split("<think>")[-1].split("</think>")[0].strip()
    ratio = round(len(think_text.split()) / max_reward_len, 2)
    if ratio > 1.0:
        ratio = 1.0

    final_reward = ratio * (max_possible_reward - min_possible_reward) + min_possible_reward
    return float(final_reward)
