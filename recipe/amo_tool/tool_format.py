
import re


def customize_format_reward_func(solution_str: str, ground_truth: str) -> float:
    """
    Compute RLLA format reward for a single example.
    Inputs:
      - solution_str: model output (raw)
      - ground_truth: ground truth string (raw)
    Output:
      - format score (float)
    """
    response = solution_str
    ans = ground_truth

    max_possible_reward = 1.0
    min_possible_reward = 0.0

    reward = min_possible_reward

    if "<response>" in ans and "<tool_call>" not in ans:
        pattern = r"^<think>.*?</think>\n<response>.*?</response>$"
        if re.search(pattern, response, re.DOTALL) and response.count("<response>") == 1 and response.count("</response>") == 1:
            reward = max_possible_reward

    elif "<response>" not in ans and "<tool_call>" in ans:
        pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$"
        if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1:
            reward = max_possible_reward

    elif "<response>" in ans and "<tool_call>" in ans:
        pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$"
        if (
            re.search(pattern, response, re.DOTALL)
            and response.count("<tool_call>") == 1
            and response.count("</tool_call>") == 1
            and response.count("<response>") == 1
            and response.count("</response>") == 1
        ):
            reward = max_possible_reward

    else:
        pattern = r"^<think>.*?</think>$"
        if re.search(pattern, response, re.DOTALL):
            reward = max_possible_reward

    return float(reward)


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """
    Compute RLLA format reward for a single example.
    Inputs:
      - solution_str: model output (raw)
      - ground_truth: ground truth string (raw)
    Output:
      - format score (float)
    """

    return customize_format_reward_func(solution_str, ground_truth)