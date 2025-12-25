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

from collections import defaultdict
from typing import Any, List

import asyncio
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.amo_vanilla import AmoVanillaRewardManager



@register("amo_hv")
class AmoHvRewardManager(AmoVanillaRewardManager):
    """Multi-objective reward manager based on hypervolume (HV) contribution.

    This manager computes per-sample rewards as the incremental contribution of
    each sample to the group-wise dominated hypervolume.

    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: dict,
        reward_fn_key: str = "data_source",
        hv_config: dict | None = None,
        **_: Any,
    ) -> None:
        """Initialize the AmoHvRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: Number of examples to print for debugging.
            compute_score: Dict of reward functions (multi-objective).
            reward_fn_key: Key used to access the data source in
                ``non_tensor_batch``. Defaults to "data_source".
            hv_config: Configuration dict for HV-based reward shaping.
        """
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key)

        # HV configuration
        hv_config = dict(hv_config) if hv_config is not None else {}
        self.hv_config: dict[str, Any] = hv_config

        # Reference point configuration
        self.reference_point_strategy: str = hv_config.get(
            "reference_point_strategy", "dynamic_batch"
        )
        # Only used when strategy == "static"
        self.reference_point = hv_config.get("reference_point", None)
        # Margin to subtract from min objective values when using dynamic reference points
        self.reference_point_margin: float = float(hv_config.get("reference_point_margin", 0.0))

        # Reward post-processing
        self.clip_negative: bool = bool(hv_config.get("clip_negative", True))
        self.reward_scaling_mode: str = hv_config.get("reward_scaling_mode", "min-max")

        # HV approximation options
        self.mc_sample_count: int = int(hv_config.get("mc_sample_count", 2048))

        # NOTE: For initial version we do *not* normalize objective vectors for HV
        # calculation to avoid changing the geometry of the dominated region.
        self.normalize_vectors_for_hv: bool = bool(
            hv_config.get("normalize_vectors_for_hv", False)
        )

        print(f"[Amo][HV] Using HV reward manager with hv_config: {self.hv_config}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:

        # If there is rm score, we directly return rm score. Otherwise, we compute
        # rewards via the multi-objective HV contribution.
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        batch_size = len(data)
        responses = data.batch["responses"]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)

        # First pass: compute individual multi-objective scores per sample and
        # gather metadata needed for group-wise HV computation.
        individual_scores_list: list[list[float]] = []
        data_sources: list[str] = []
        uids: list[str] = []
        valid_response_lengths: list[int] = []
        prompt_strs: list[str] = []
        response_strs: list[str] = []
        ground_truths: list[Any] = []

        already_print_data_sources: dict[str, int] = {}

        for i in range(batch_size):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            # [Amo] compute individual scores (one vector per sample)
            single_run_item = asyncio.run(
                self.compute_individual_reward(
                    data_source=data_source,
                    response_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            )

            individual_scores = single_run_item["individual_scores"]
            reward_extra_info_item = single_run_item["reward_extra_info"]
            for key, value in reward_extra_info_item.items():
                reward_extra_info[key].append(value)

            # Store for HV computation
            individual_scores_list.append([float(s) for s in individual_scores])
            data_sources.append(data_source)
            
            # Get stable uid for grouping
            uid = data_item.non_tensor_batch.get("uid")
            assert uid is not None, "uid should not be None"
            
            uids.append(uid)
            valid_response_lengths.append(valid_response_length)
            prompt_strs.append(prompt_str)
            response_strs.append(response_str)
            ground_truths.append(ground_truth)

        if not individual_scores_list:
            # Empty batch or no objectives; return zero reward tensor.
            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return reward_tensor

        # ------------------------------------------------------------------
        # Group-wise HV computation
        # ------------------------------------------------------------------
        score_tensor = torch.tensor(individual_scores_list, dtype=torch.float32)
        dim = score_tensor.shape[1]

        # Prepare reference points for dynamic-datasource strategy if needed
        datasource_ref_points: dict[str, torch.Tensor] = {}
        if self.reference_point_strategy == "dynamic_datasource":
            ds2indices: dict[str, list[int]] = defaultdict(list)
            for idx, ds in enumerate(data_sources):
                ds2indices[ds].append(idx)
            for ds, indices in ds2indices.items():
                vals = score_tensor[indices]
                min_vals = vals.min(dim=0).values
                r = min_vals - self.reference_point_margin
                datasource_ref_points[ds] = r

        # Prepare static reference point if configured
        static_ref_point: torch.Tensor | None = None
        if self.reference_point_strategy == "static":
            if self.reference_point is None:
                raise ValueError(
                    "[Amo][HV] reference_point_strategy is 'static' but 'reference_point' is not provided."
                )
            static_ref_point = torch.tensor(self.reference_point, dtype=torch.float32)
            if static_ref_point.numel() != dim:
                raise ValueError(
                    f"[Amo][HV] reference_point dimension mismatch: got {static_ref_point.numel()}, expected {dim}."
                )

        # Group indices by uid
        uid2indices: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            uid2indices[uid].append(idx)

        hv_contributions = torch.zeros(len(uids), dtype=torch.float32)
        total_hv = torch.zeros(len(uids), dtype=torch.float32)
        reference_points_per_sample: list[list[float]] = [
            [0.0 for _ in range(dim)] for _ in range(len(uids))
        ]
        group_sizes = [0 for _ in range(len(uids))]
        print(f"[Amo][HV] uid2indices: {uid2indices}")

        for group_uid, indices in uid2indices.items():
            group_scores = score_tensor[indices]  # (group_size, dim)
            if group_scores.numel() == 0:
                continue

            # Determine reference point for this group
            if self.reference_point_strategy == "dynamic_batch":
                group_min = group_scores.min(dim=0).values
                ref_point = group_min - self.reference_point_margin
            elif self.reference_point_strategy == "static":
                assert static_ref_point is not None
                ref_point = static_ref_point
            elif self.reference_point_strategy == "dynamic_datasource":
                ds = data_sources[indices[0]]
                ref_point = datasource_ref_points[ds]
            else:
                raise ValueError(
                    f"[Amo][HV] Unsupported reference_point_strategy: {self.reference_point_strategy}"
                )

            # Ensure reference point is dominated by all objective vectors in this group
            group_min = group_scores.min(dim=0).values
            ref_point = torch.minimum(ref_point, group_min)

            # Optional vector-level normalization for HV.
            # In the initial version, we intentionally keep the objective
            # vectors unchanged to avoid altering HV geometry. The flag
            # ``normalize_vectors_for_hv`` is reserved for future use.
            hv_vectors = group_scores
            if self.normalize_vectors_for_hv:
                # NOTE: No-op by design in the initial implementation.
                pass

            # Compute group-wise HV and per-sample HV without each point
            group_hv, hv_without_each = self._compute_group_hv(hv_vectors, ref_point)
            contributions = group_hv - hv_without_each  # (group_size,)

            # Post-process contributions
            if self.clip_negative:
                contributions = torch.clamp(contributions, min=0.0)

            contributions = self._scale_contributions(contributions, self.reward_scaling_mode)

            # Write rewards to the last token position and fill extra info
            for local_idx, global_idx in enumerate(indices):
                hv_contributions[global_idx] = contributions[local_idx]
                total_hv[global_idx] = group_hv
                reference_points_per_sample[global_idx] = ref_point.tolist()
                group_sizes[global_idx] = len(indices)

                valid_response_length = valid_response_lengths[global_idx]
                if valid_response_length > 0:
                    reward_tensor[global_idx, valid_response_length - 1] = contributions[local_idx]

        # ------------------------------------------------------------------
        # Debug printing (keep original behavior controlled by num_examine)
        # ------------------------------------------------------------------
        for i in range(batch_size):
            data_source = data_sources[i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_strs[i])
                print("[response]", response_strs[i])
                print("[ground_truth]", ground_truths[i])

                # Print individual scores for interpretability
                for reward_fn_name, score in zip(self.compute_score.keys(), individual_scores_list[i]):
                    print(f"[{reward_fn_name} score]", score)
                print("[hv_contribution]", hv_contributions[i].item())
                print("[total_hv]", total_hv[i].item())

        # Attach HV-related extra information (aligned with sample order)
        for i in range(batch_size):
            reward_extra_info["hv_contribution"].append(hv_contributions[i].item())
            reward_extra_info["total_hv"].append(total_hv[i].item())
            reward_extra_info["reference_point"].append(reference_points_per_sample[i])
            reward_extra_info["group_uid"].append(uids[i])
            reward_extra_info["dim"].append(dim)
            reward_extra_info["group_size"].append(group_sizes[i])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_vectors_for_hv(
        vectors: torch.Tensor,
        ref_point: torch.Tensor,
    ) -> torch.Tensor:
        """Optionally normalize objective vectors for HV.

        This is kept for extensibility, but disabled by default because changing
        the geometry of the dominated region will alter HV values in ways that
        may be undesirable. When enabled, we perform a simple per-dimension
        min-max normalization within the group.
        """
        if vectors.numel() == 0:
            return vectors

        v_min = vectors.min(dim=0).values
        v_max = vectors.max(dim=0).values
        range_ = torch.clamp(v_max - v_min, min=1e-8)
        normalized = (vectors - v_min) / range_

        # Adjust reference point to the same scale if needed
        # (assumes ref_point <= v_min component-wise)
        return normalized

    @staticmethod
    def _scale_contributions(contribs: torch.Tensor, mode: str) -> torch.Tensor:
        """Scale HV contributions within a group.

        Only the *contributions* are scaled; the objective vectors themselves
        are left untouched to avoid affecting HV geometry.
        """
        if contribs.numel() == 0:
            return contribs

        if mode == "none":
            return contribs
        if mode == "min-max":
            c_min = contribs.min()
            c_max = contribs.max()
            if (c_max - c_min) <= 0:
                return torch.zeros_like(contribs)
            return (contribs - c_min) / (c_max - c_min + 1e-8)
        if mode == "z-score":
            mean = contribs.mean()
            std = contribs.std(unbiased=False)
            if std <= 0:
                return contribs - mean
            return (contribs - mean) / (std + 1e-8)
        if mode == "tanh":
            return 0.5 + 1.5 * torch.tanh(contribs)

        raise ValueError(f"[Amo][HV] Unsupported reward_scaling_mode: {mode}")

    @staticmethod
    def _remove_index(vectors: torch.Tensor, index: int) -> torch.Tensor:
        """Create a new tensor with the row at ``index`` removed."""
        if vectors.shape[0] <= 1:
            return vectors.new_zeros((0, vectors.shape[1]))
        return torch.cat([vectors[:index], vectors[index + 1 :]], dim=0)

    def _compute_group_hv(
        self,
        vectors: torch.Tensor,
        ref_point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute group HV and HV without each point.

        Args:
            vectors: Tensor of shape (group_size, dim).
            ref_point: Reference point tensor of shape (dim,).

        Returns:
            A tuple ``(hv_total, hv_without_each)`` where:
            - hv_total: scalar tensor, HV(S, ref_point).
            - hv_without_each: tensor of shape (group_size,), where the i-th
              element is HV(S - {i}, ref_point).
        """
        group_size, dim = vectors.shape

        if group_size == 0 or dim == 0:
            return vectors.new_tensor(0.0), vectors.new_zeros(group_size)

        # hv_total = self._hv_recursive_slicing(vectors.tolist(), ref_point.tolist())
        hv_total = self._hv_recursive_slicing(vectors, ref_point)
        hv_without_each = []
        for i in range(group_size):
            sub = self._remove_index(vectors, i)
            hv_without_each.append(self._hv_recursive_slicing(sub, ref_point))

        hv_without_each_tensor = torch.stack(hv_without_each) if hv_without_each else vectors.new_zeros(0)
        return hv_total, hv_without_each_tensor

    def _hv_recursive_slicing(
        self,
        points: torch.Tensor,
        ref_point: torch.Tensor,
    ) -> float:
        """Compute hypervolume using recursive slicing algorithm."""
        hv: float = self._hv_recursive_slicing_helper(
            points.tolist(), 
            ref_point.tolist()
        )
        return torch.tensor(hv, dtype=ref_point.dtype, device=ref_point.device).clamp(min=0.0)

    def _hv_recursive_slicing_helper(
        self,
        points: List[tuple],
        ref_point: tuple,
    ) -> float:
        """
        Adapted from Fonseca et al. (2006) recursive slicing algorithm.
        """
        if not points:
            return 0.0

        # 1-D case
        if len(ref_point) == 1:
            return max(p[0] for p in points) - ref_point[0]

        points = sorted(points, key=lambda p: p[0])

        hv = 0.0
        ref0 = ref_point[0] # moving slice position
        while points:
            # Current slice: width along dim-0
            p0 = points[0]
            width = p0[0] - ref0
            if width > 0:
                # All points in this slice, projected to the remaining m-1 dims
                slice_pts = [p[1:] for p in points]
                slice_ref = ref_point[1:]
                hv += width * self._hv_recursive_slicing_helper(slice_pts, slice_ref)
                ref0 = p0[0]
            # Keep only points strictly beyond the present slice
            points = [p for p in points if p[0] > p0[0]]

        return hv