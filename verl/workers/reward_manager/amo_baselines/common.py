# Copyright 2025 Rihong Qiu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared machinery for multi-objective *baseline* reward managers.

The per-sample multi-objective scoring loop is identical across every Amo
reward manager (vanilla, HVPO and every baseline in this subpackage): decode
the response, run all reward functions, collect an ``(batch, num_objectives)``
score matrix plus the ``uid`` grouping. Only the mapping from that matrix to a
scalar per-response reward changes.

:class:`AmoBaselineRewardManager` captures the common part once and exposes a
single hook, :meth:`_compute_scalar_rewards`, that subclasses implement to turn
the score matrix into a ``(batch,)`` scalar-reward vector. This guarantees the
baselines share the *exact* scoring / grouping / bookkeeping code with HVPO, so
the only experimental variable is the multi-objective credit rule.

It also provides normalization and scalarization helpers (linear, Tchebycheff,
group-wise z-scoring) reused by several managers.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager.amo_vanilla import AmoVanillaRewardManager


# ----------------------------------------------------------------------
# Normalization / scalarization helpers (pure functions, unit-tested)
# ----------------------------------------------------------------------
def normalize_weights(weights, num_objectives: int) -> torch.Tensor:
    """Return a non-negative weight vector summing to 1 of length ``num_objectives``.

    ``None`` yields uniform weights. A shorter/longer list is an error, so the
    caller cannot silently mis-specify an objective ordering.
    """
    if weights is None:
        return torch.full((num_objectives,), 1.0 / num_objectives, dtype=torch.float32)
    w = torch.tensor([float(x) for x in weights], dtype=torch.float32)
    if w.numel() != num_objectives:
        raise ValueError(
            f"[Amo][baseline] weights length {w.numel()} != num_objectives {num_objectives}"
        )
    if torch.any(w < 0):
        raise ValueError(f"[Amo][baseline] weights must be non-negative, got {weights}")
    total = w.sum()
    if total <= 0:
        raise ValueError(f"[Amo][baseline] weights must sum to a positive value, got {weights}")
    return w / total


def linear_scalarize(scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted sum ``sum_j w_j r_j`` over objectives.

    Args:
        scores: ``(batch, num_objectives)`` score matrix.
        weights: ``(num_objectives,)`` weight vector.

    Returns:
        ``(batch,)`` scalar rewards.
    """
    return scores @ weights.to(scores.dtype)


def tchebycheff_scalarize(
    scores: torch.Tensor,
    weights: torch.Tensor,
    ideal: torch.Tensor,
    rho: float = 0.0,
) -> torch.Tensor:
    """(Augmented) Tchebycheff scalarization for *maximization*.

    ``r_Tche = -max_j w_j (z*_j - r_j) + rho * sum_j w_j r_j``

    where ``z*`` is the (fixed) ideal point. ``rho == 0`` gives the plain
    Tchebycheff scalarizer; a small ``rho > 0`` (augmented form) suppresses
    weakly-Pareto-optimal solutions and ties. Unlike linear scalarization, this
    can reach concave regions of the Pareto front.

    Args:
        scores: ``(batch, num_objectives)`` score matrix.
        weights: ``(num_objectives,)`` weight vector.
        ideal: ``(num_objectives,)`` fixed ideal (reference/utopia) point.
        rho: augmentation coefficient (>= 0).

    Returns:
        ``(batch,)`` scalar rewards (higher is better).
    """
    w = weights.to(scores.dtype)
    ideal = ideal.to(scores.dtype)
    # weighted gap to the ideal point per objective; max over objectives.
    gap = w * (ideal.unsqueeze(0) - scores)  # (batch, m)
    tche = -gap.max(dim=1).values  # (batch,)
    if rho and rho != 0.0:
        tche = tche + float(rho) * (scores @ w)
    return tche


def group_zscore(
    scores: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Per-objective, per-group z-score (the GDPO within-group standardization).

    For each ``uid`` group and each objective column, subtract the group mean
    and divide by the group std. Returns a matrix of the same shape as
    ``scores``. Singleton groups get a zero-centered, unit-scaled value.

    Args:
        scores: ``(batch, num_objectives)`` score matrix.
        index: length-``batch`` array of group ids (uids).
        epsilon: numerical floor for the std.

    Returns:
        ``(batch, num_objectives)`` standardized scores.
    """
    out = torch.zeros_like(scores)
    id2rows: dict[Any, list[int]] = defaultdict(list)
    for i in range(scores.shape[0]):
        id2rows[index[i]].append(i)
    for _uid, rows in id2rows.items():
        idx = torch.tensor(rows, dtype=torch.long, device=scores.device)
        block = scores.index_select(0, idx)  # (g, m)
        if block.shape[0] == 1:
            out.index_copy_(0, idx, torch.zeros_like(block))
            continue
        mean = block.mean(dim=0, keepdim=True)
        std = block.std(dim=0, unbiased=False, keepdim=True)
        out.index_copy_(0, idx, (block - mean) / (std + epsilon))
    return out


def group_indices(index: np.ndarray) -> dict[Any, list[int]]:
    """Map each group id to the list of row indices belonging to it."""
    id2rows: dict[Any, list[int]] = defaultdict(list)
    for i in range(len(index)):
        id2rows[index[i]].append(i)
    return id2rows


# ----------------------------------------------------------------------
# Base reward manager
# ----------------------------------------------------------------------
class AmoBaselineRewardManager(AmoVanillaRewardManager):
    """Base class for multi-objective baseline reward managers.

    Subclasses implement :meth:`_compute_scalar_rewards`, receiving the full
    ``(batch, num_objectives)`` score matrix and the per-sample ``uid`` grouping
    and returning a ``(batch,)`` scalar-reward vector. Everything else -- reward
    function evaluation, decoding, ``rm_scores`` passthrough, writing the reward
    at the final response token, debug printing and per-objective bookkeeping --
    is inherited from here.

    The manager also always emits ``token_level_scores_dict`` (one
    last-token-placed tensor per objective) so GDPO-family advantage estimators
    can consume per-objective signals when paired with such a manager.
    """

    #: Whether ``_compute_scalar_rewards`` should be called with grad disabled.
    _no_grad_rewards = True

    def _compute_scalar_rewards(
        self,
        score_tensor: torch.Tensor,
        uids: np.ndarray,
        is_train: bool,
        extra: dict[str, Any],
    ) -> torch.Tensor:
        """Map the ``(batch, m)`` score matrix to ``(batch,)`` scalar rewards.

        Args:
            score_tensor: ``(batch, num_objectives)`` per-sample objective scores.
            uids: length-``batch`` array of prompt-group ids.
            is_train: True for training rollouts, False for validation/test.
            extra: dict with side info (currently ``{"reward_extra_info": ...}``)
                that a subclass may augment for logging.

        Returns:
            ``(batch,)`` scalar-reward vector.
        """
        raise NotImplementedError

    def _validation_reward(self, score_tensor: torch.Tensor) -> torch.Tensor:
        """Interpretable scalar for validation/test logging.

        Defaults to the mean over objectives, matching HVPO's monitoring signal
        so val curves are comparable across methods. Subclasses may override
        (e.g. a constrained method may want a satisfaction-aware scalar) but this
        value is *only* used for logging, never as a training gradient.
        """
        return score_tensor.mean(dim=1)

    # ------------------------------------------------------------------
    def __call__(self, data: DataProto, return_dict: bool = False):
        # rm_scores passthrough (identical to base managers).
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        batch_size = len(data)
        responses = data.batch["responses"]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)

        objective_names = list(self.compute_score.keys())
        token_level_scores_dict = {
            name: torch.zeros_like(responses, dtype=torch.float32) for name in objective_names
        }

        individual_scores_list: list[list[float]] = []
        data_sources: list[str] = []
        uids: list[Any] = []
        data_splits: list[str] = []
        valid_response_lengths: list[int] = []
        prompt_strs: list[str] = []
        response_strs: list[str] = []
        ground_truths: list[Any] = []

        for i in range(batch_size):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            single_run_item = asyncio.run(
                self.compute_individual_reward(
                    data_source=data_source,
                    response_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            )
            individual_scores = single_run_item["individual_scores"]
            for key, value in single_run_item["reward_extra_info"].items():
                reward_extra_info[key].append(value)

            individual_scores_list.append([float(s) for s in individual_scores])
            data_sources.append(data_source)

            split = extra_info.get("split")
            assert split in ["train", "val", "test"], f"split should be train/val/test, got {split}"
            uid = data_item.non_tensor_batch.get("uid")
            assert uid is not None, "uid should not be None"

            uids.append(uid)
            data_splits.append(split)
            valid_response_lengths.append(valid_response_length)
            prompt_strs.append(prompt_str)
            response_strs.append(response_str)
            ground_truths.append(ground_truth)

        if not individual_scores_list:
            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            return reward_tensor

        score_tensor = torch.tensor(individual_scores_list, dtype=torch.float32)
        assert all(s == data_splits[0] for s in data_splits), "mixed splits in one batch"
        is_train = data_splits[0] == "train"
        uid_array = np.array(uids, dtype=object)

        extra: dict[str, Any] = {"reward_extra_info": reward_extra_info}
        ctx = torch.no_grad() if self._no_grad_rewards else _nullcontext()
        with ctx:
            if is_train:
                scalar_rewards = self._compute_scalar_rewards(score_tensor, uid_array, True, extra)
            else:
                scalar_rewards = self._validation_reward(score_tensor)
        scalar_rewards = scalar_rewards.to(torch.float32)
        assert scalar_rewards.shape == (batch_size,), (
            f"[Amo][baseline] scalar reward shape {scalar_rewards.shape} != ({batch_size},)"
        )

        already_print: dict[str, int] = {}
        for i in range(batch_size):
            valid_response_length = valid_response_lengths[i]
            assert valid_response_length > 0, f"valid_response_lengths[{i}] = {valid_response_length}"
            reward_tensor[i, valid_response_length - 1] = scalar_rewards[i]
            for j, name in enumerate(objective_names):
                token_level_scores_dict[name][i, valid_response_length - 1] = score_tensor[i, j]

            reward_extra_info["scalar_reward"].append(float(scalar_rewards[i]))

            data_source = data_sources[i]
            already_print.setdefault(data_source, 0)
            if already_print[data_source] < self.num_examine:
                already_print[data_source] += 1
                print("[prompt]", prompt_strs[i])
                print("[response]", response_strs[i])
                print("[ground_truth]", ground_truths[i])
                for name, s in zip(objective_names, individual_scores_list[i]):
                    print(f"[{name} score]", s)
                print("[scalar_reward]", float(scalar_rewards[i]))

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "token_level_scores_dict": token_level_scores_dict,
            }
        return reward_tensor


class _nullcontext:
    """Minimal no-op context manager (Python 3.7+ has contextlib.nullcontext,
    kept local to avoid an import and to make the grad toggle explicit)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
