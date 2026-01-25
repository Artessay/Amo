
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
    half_possible_reward = (max_possible_reward + min_possible_reward) / 2.0

    reward = min_possible_reward

    # Common <think> block (allow arbitrary whitespace)
    think_pattern = r"<think>\s*.*?\s*</think>"

    # <response> / <tool_call> blocks (do not require them to be on separate lines)
    response_pattern = r"<response>\s*.*?\s*</response>"
    tool_call_pattern = r"<tool_call>\s*.*?\s*</tool_call>"

    # Case 1: ground truth has <response> only
    if "<response>" in ans and "<tool_call>" not in ans:
        pattern = rf"^\s*{think_pattern}\s*{response_pattern}\s*$"
        if (
            re.search(pattern, response, re.DOTALL)
            and response.count("<response>") == 1
            and response.count("</response>") == 1
        ):
            reward = max_possible_reward

    # Case 2: ground truth has <tool_call> only
    elif "<response>" not in ans and "<tool_call>" in ans:
        pattern = rf"^\s*{think_pattern}\s*{tool_call_pattern}\s*$"
        if (
            re.search(pattern, response, re.DOTALL)
            and response.count("<tool_call>") == 1
            and response.count("</tool_call>") == 1
        ):
            reward = max_possible_reward

    # Case 3: ground truth has both <tool_call> and <response>
    elif "<response>" in ans and "<tool_call>" in ans:
        pattern = rf"^\s*{think_pattern}\s*{tool_call_pattern}\s*{response_pattern}\s*$"
        if (
            re.search(pattern, response, re.DOTALL)
            and response.count("<tool_call>") == 1
            and response.count("</tool_call>") == 1
            and response.count("<response>") == 1
            and response.count("</response>") == 1
        ):
            reward = max_possible_reward

    # Case 4: ground truth has neither <tool_call> nor <response>
    else:
        # Half credit if output contains ONLY ONE of the blocks: <think> OR <tool_call> OR <response>
        only_think_pattern = rf"^\s*{think_pattern}\s*$"
        only_tool_call_pattern = rf"^\s*{tool_call_pattern}\s*$"
        only_response_pattern = rf"^\s*{response_pattern}\s*$"

        has_only_think = (
            re.search(only_think_pattern, response, re.DOTALL)
            and response.count("<think>") == 1
            and response.count("</think>") == 1
        )

        has_only_tool_call = (
            re.search(only_tool_call_pattern, response, re.DOTALL)
            and response.count("<tool_call>") == 1
            and response.count("</tool_call>") == 1
        )
        has_only_response = (
            re.search(only_response_pattern, response, re.DOTALL)
            and response.count("<response>") == 1
            and response.count("</response>") == 1
        )

        # Half credit if it starts with a well-formed <think>...</think> block
        # but the remaining content does NOT satisfy the expected overall format.
        starts_with_think_block = bool(
            re.search(rf"^\s*{think_pattern}", response, re.DOTALL)
        )

        if has_only_think or has_only_tool_call or has_only_response or starts_with_think_block:
            reward = half_possible_reward

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
