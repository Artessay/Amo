"""Paper-faithful, response-wise Hypervolume-Guided Policy Optimization."""

import asyncio
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.amo_vanilla import AmoVanillaRewardManager


def compute_hvpo_credit(raw_scores, lower, upper, reference):
    """Return Eq. (14) credit, calibrated coordinates, and shortfall."""
    if raw_scores.ndim != 2:
        raise ValueError(f"raw_scores must be 2-D, got {raw_scores.shape}")
    span = upper - lower
    if torch.any(span <= 0):
        raise ValueError("every HVPO upper anchor must exceed its lower anchor")
    z = (raw_scores - lower) / span
    gains = z - reference
    acceptable = torch.all(gains >= 0, dim=-1)
    positive = torch.all(gains > 0, dim=-1)
    rooted_hv = raw_scores.new_zeros(raw_scores.shape[0])
    # No epsilon: a point on the reference boundary must receive exactly zero.
    rooted_hv[positive] = torch.exp(torch.log(gains[positive]).mean(dim=-1))
    shortfall = torch.clamp(reference - z, min=0).amax(dim=-1)
    return torch.where(acceptable, rooted_hv, -shortfall), z, shortfall


@register("amo_hvpo")
class AmoHvpoRewardManager(AmoVanillaRewardManager):
    """Assign each response an independent rooted singleton-HV credit."""

    def __init__(self, tokenizer, num_examine: int, compute_score: dict,
                 reward_fn_key: str = "data_source", hv_config: dict | None = None,
                 **_: Any) -> None:
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key)
        cfg = dict(hv_config or {})
        count = len(compute_score)
        lower, upper = cfg.get("calib_lower"), cfg.get("calib_upper")
        reference = cfg.get("reference_point", [0.0] * count)
        if lower is None or upper is None:
            raise ValueError("HVPO requires frozen calib_lower and calib_upper anchors")
        for name, value in (("calib_lower", lower), ("calib_upper", upper),
                            ("reference_point", reference)):
            if len(value) != count:
                raise ValueError(f"HVPO {name} has length {len(value)}; expected {count}")
        self.calib_lower = torch.tensor(lower, dtype=torch.float32)
        self.calib_upper = torch.tensor(upper, dtype=torch.float32)
        self.reference_point = torch.tensor(reference, dtype=torch.float32)
        if torch.any(self.calib_upper <= self.calib_lower):
            raise ValueError("every HVPO upper anchor must exceed its lower anchor")
        print(f"[Amo][HVPO] singleton HV: lower={lower}, upper={upper}, reference={reference}")

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch:
            raise ValueError("amo_hvpo requires per-objective rewards, not scalar rm_scores")
        responses = data.batch["responses"]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        extra: dict[str, list] = defaultdict(list)
        rows, lengths, debug = [], [], []
        for i in range(len(data)):
            item = data[i]
            prompts = item.batch["prompts"]
            prompt_len = prompts.shape[-1]
            valid_prompt = int(item.batch["attention_mask"][:prompt_len].sum())
            valid_response = int(item.batch["attention_mask"][prompt_len:].sum())
            if valid_response <= 0:
                raise ValueError(f"HVPO response {i} has no valid tokens")
            prompt = self.tokenizer.decode(prompts[-valid_prompt:], skip_special_tokens=True)
            response = self.tokenizer.decode(item.batch["responses"][:valid_response], skip_special_tokens=True)
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            source = item.non_tensor_batch[self.reward_fn_key]
            info = item.non_tensor_batch.get("extra_info", {})
            info["num_turns"] = item.non_tensor_batch.get("__num_turns__")
            info["rollout_reward_scores"] = item.non_tensor_batch.get("reward_scores", {})
            result = asyncio.run(self.compute_individual_reward(
                data_source=source, response_str=response, ground_truth=ground_truth,
                extra_info=info))
            scores = [float(x) for x in result["individual_scores"]]
            if len(scores) != len(self.compute_score):
                raise ValueError(f"HVPO scorer returned {len(scores)} objectives; expected {len(self.compute_score)}")
            rows.append(scores)
            lengths.append(valid_response)
            debug.append((prompt, response, ground_truth, source))
            for key, value in result["reward_extra_info"].items():
                extra[key].append(value)
        if not rows:
            output = {"reward_tensor": reward_tensor, "reward_extra_info": extra}
            return output if return_dict else reward_tensor
        raw = torch.tensor(rows, dtype=torch.float32)
        credit, calibrated, shortfall = compute_hvpo_credit(
            raw, self.calib_lower.to(raw), self.calib_upper.to(raw), self.reference_point.to(raw))
        printed = defaultdict(int)
        for i in range(len(rows)):
            reward_tensor[i, lengths[i] - 1] = credit[i]
            extra["hvpo_credit"].append(float(credit[i]))
            extra["hvpo_shortfall"].append(float(shortfall[i]))
            extra["hvpo_calibrated_scores"].append(calibrated[i].tolist())
            extra["hvpo_raw_scores"].append(raw[i].tolist())
            prompt, response, truth, source = debug[i]
            if printed[source] < self.num_examine:
                printed[source] += 1
                print("[prompt]", prompt)
                print("[response]", response)
                print("[ground_truth]", truth)
                print("[raw objectives]", raw[i].tolist())
                print("[calibrated objectives]", calibrated[i].tolist())
                print("[hvpo credit]", float(credit[i]))
        output = {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return output if return_dict else reward_tensor
