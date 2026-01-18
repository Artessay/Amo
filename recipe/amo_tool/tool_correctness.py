# rlla_tool_correctness_score.py
import json
import os
from collections import Counter



def match_score(list1, list2) -> float:
    """Compute a similarity score considering element frequency, ignoring order."""
    if list1 == list2:
        return 1.0

    if os.getenv("REFINEDREWARD", 0) == "1":
        # strict match
        return 1.0 if list1 == list2 else 0.0

    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)
    count2 = Counter(list2)
    intersection = sum(min(count1[k], count2[k]) for k in (count1.keys() & count2.keys()))
    max_possible = len(list1) + len(list2) - intersection
    return intersection / max_possible if max_possible > 0 else 0.0


def compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward: float, min_possible_reward: float) -> float:
    if gt_tools == pd_tools:
        return max_possible_reward

    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))

    local_max_possible = 1.0
    used_pd_indices = set()

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]

        if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
            local_max_possible += 1.0
        else:
            local_max_possible += 1.0 + len(gt_params)

        best_match_score = 0.0
        best_match_index = -1

        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool.get("name") != gt_name:
                continue

            if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
                if gt_tool == pd_tool:
                    best_match_score = 1.0
                    best_match_index = i
                    break
                else:
                    continue

            pd_params = pd_tool.get("parameters", {})
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
            correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

            total_score = param_score + correctness_score
            if total_score > best_match_score:
                best_match_score = total_score
                best_match_index = i

        if best_match_index != -1:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    # normalize to [min_possible_reward, max_possible_reward]
    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward


def customize_correctness_reward_tool(solution_str: str, ground_truth: str) -> float:
    """
    Compute RLLA tool-call correctness reward for a single example.
    If ground_truth has no <tool_call>, returns 0.0 (same as original code).
    """
    response = solution_str
    ans = ground_truth

    # tool reward range
    tool_max_possible = 3.0
    tool_min_possible = -3.0

    # If GT has no tool call, reward is 0.0 in the original code
    if "<tool_call>" not in ans:
        return 0.0

    # parse GT tools (each line is a JSON dict)
    gt_tool_call = ans.split("<tool_call>")[1].split("</tool_call>")[0].strip()
    gt_tools = [json.loads(line) for line in gt_tool_call.split("\n") if line.strip()]

    try:
        assert "<tool_call>" in response and "</tool_call>" in response
        pd_block = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        pd_tools = [json.loads(line) for line in pd_block.split("\n") if line.strip()]
        reward = compute_tool_call_reward(gt_tools, pd_tools, tool_max_possible, tool_min_possible)
    except Exception:
        reward = tool_min_possible

    return float(reward)


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """
    Compute RLLA tool-call correctness reward for a single example.
    Inputs:
      - solution_str: model output (raw)
      - ground_truth: ground truth string (raw)
    Output:
      - format score (float)
    """
    
    return customize_correctness_reward_tool(solution_str, ground_truth)