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
import threading
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.amo_vanilla import AmoVanillaRewardManager


@register("amo_hv")
class AmoHvRewardManager(AmoVanillaRewardManager):
    """Multi-objective reward manager based on hypervolume (HV) contribution.

    This manager computes per-sample rewards as the incremental contribution of
    each sample to the dominated hypervolume.

    By default, contributions are computed within each ``uid`` group only
    (intra-group HV). When ``hv_config['use_global_pareto_cache']`` is enabled,
    the manager additionally maintains a bounded-size global Pareto front of
    objective vectors and computes rewards as::

        ΔHV(i) = HV(P ∪ {v_i}, r) - HV(P, r),

    where ``P`` is the global Pareto cache and ``r`` is a reference point
    chosen according to ``hv_config['reference_point_strategy']``:

    * ``"dynamic_batch"``: ``r`` is built from the union of the global cache
      and the current group's vectors (min per-dimension minus an optional
      margin, then clamped so it is dominated by all group points).
    * ``"static"``: ``r`` is the user-provided ``hv_config['reference_point']``.

    The cache behaviour is controlled via:

    * ``use_global_pareto_cache`` (bool): enable/disable global mode.
    * ``pareto_cache_max_size`` (int): maximum number of stored Pareto points.
    * ``pareto_cache_eps`` (float): dominance tolerance.
    * ``pareto_cache_strategy`` (str): eviction strategy (currently only
      ``"fifo"``, keeping the most recent non-dominated points).
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

        # Global Pareto front cache configuration
        self.use_global_pareto_cache: bool = bool(
            hv_config.get("use_global_pareto_cache", False)
        )
        self.pareto_cache_max_size: int = int(hv_config.get("pareto_cache_max_size", 1024))
        self.pareto_cache_eps: float = float(hv_config.get("pareto_cache_eps", 1e-9))
        self.pareto_cache_strategy: str = hv_config.get("pareto_cache_strategy", "fifo")
        if self.pareto_cache_strategy not in {"fifo"}:
            raise ValueError(
                f"[Amo][HV] Unsupported pareto_cache_strategy: {self.pareto_cache_strategy}"
            )

        # Internal global Pareto cache state (objective vectors only).
        # The cache stores a bounded set of non-dominated points under maximization.
        self._pareto_cache: list[list[float]] = []
        self._pareto_lock = threading.Lock()
        self._pareto_dim: int | None = None

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
        # Group-wise / global HV computation
        # ------------------------------------------------------------------
        score_tensor = torch.tensor(individual_scores_list, dtype=torch.float32)
        dim = score_tensor.shape[1]

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
        cache_sizes = [0 for _ in range(len(uids))]
        used_global_cache_flags = [False for _ in range(len(uids))]

        # Take a snapshot of the global Pareto cache for this batch. The snapshot
        # is read-only while computing rewards so that all samples in the batch
        # see a consistent frontier.
        pareto_cache_snapshot: list[list[float]] = []
        if self.use_global_pareto_cache and self.pareto_cache_max_size > 0:
            with self._pareto_lock:
                pareto_cache_snapshot = [p[:] for p in self._pareto_cache]
        use_global_cache_for_batch = bool(self.use_global_pareto_cache and self.pareto_cache_max_size > 0)
        cache_size_for_batch = len(pareto_cache_snapshot) if use_global_cache_for_batch else 0

        for group_uid, indices in uid2indices.items():
            group_scores = score_tensor[indices]  # (group_size, dim)
            if group_scores.numel() == 0:
                continue

            # Prepare Pareto cache tensor for this group (if enabled).
            if use_global_cache_for_batch and pareto_cache_snapshot:
                pareto_tensor = torch.tensor(
                    pareto_cache_snapshot,
                    dtype=group_scores.dtype,
                    device=group_scores.device,
                )
                if pareto_tensor.shape[1] != group_scores.shape[1]:
                    raise ValueError(
                        f"[Amo][HV] Pareto cache dimension mismatch: got {pareto_tensor.shape[1]}, expected {group_scores.shape[1]}."
                    )
            else:
                pareto_tensor = group_scores.new_zeros((0, group_scores.shape[1]))

            # Determine reference point for this group
            group_min = group_scores.min(dim=0).values
            if self.reference_point_strategy == "dynamic_batch":
                if use_global_cache_for_batch and pareto_tensor.numel() > 0:
                    # Use the union of the global Pareto frontier and the current group's
                    # objective vectors to determine the reference point, then clamp so
                    # that it is dominated by all group points.
                    all_points = torch.cat([group_scores, pareto_tensor], dim=0)
                    union_min = all_points.min(dim=0).values
                    ref_point = union_min - self.reference_point_margin
                else:
                    ref_point = group_min - self.reference_point_margin
            elif self.reference_point_strategy == "static":
                assert static_ref_point is not None
                ref_point = static_ref_point.to(dtype=group_scores.dtype, device=group_scores.device)
            else:
                raise ValueError(
                    f"[Amo][HV] Unsupported reference_point_strategy: {self.reference_point_strategy}"
                )

            # Ensure reference point is dominated by all objective vectors in this group
            ref_point = torch.minimum(ref_point, group_min)

            # Optional vector-level normalization for HV.
            # In the initial version, we intentionally keep the objective
            # vectors unchanged to avoid altering HV geometry. The flag
            # ``normalize_vectors_for_hv`` is reserved for future use.
            hv_vectors = group_scores
            if self.normalize_vectors_for_hv:
                # NOTE: No-op by design in the current implementation.
                pass

            # Compute HV contributions: either group-wise or against the global Pareto cache.
            if use_global_cache_for_batch:
                group_hv, contributions = self._compute_global_hv_contributions(
                    hv_vectors, ref_point, pareto_tensor
                )
            else:
                group_hv, hv_without_each = self._compute_group_hv(hv_vectors, ref_point)
                contributions = group_hv - hv_without_each  # (group_size,)
                assert contributions.shape == (len(indices),)

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
                cache_sizes[global_idx] = cache_size_for_batch
                used_global_cache_flags[global_idx] = use_global_cache_for_batch

                valid_response_length = valid_response_lengths[global_idx]
                if valid_response_length > 0:
                    reward_tensor[global_idx, valid_response_length - 1] = contributions[local_idx]

        # After computing rewards for the whole batch, update the global
        # Pareto cache once using all objective vectors from this batch. This
        # ensures that ΔHV is always measured against the cache state prior to
        # the current batch.
        if use_global_cache_for_batch:
            with self._pareto_lock:
                self._update_pareto_cache(individual_scores_list)

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
            reward_extra_info["cache_size"].append(cache_sizes[i])
            reward_extra_info["used_global_cache"].append(bool(used_global_cache_flags[i]))

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

        hv_total = self._hv_recursive_slicing(vectors, ref_point)
        hv_without_each = []
        for i in range(group_size):
            sub = self._remove_index(vectors, i)
            hv_without_each.append(self._hv_recursive_slicing(sub, ref_point))

        hv_without_each_tensor = torch.stack(hv_without_each) if hv_without_each else vectors.new_zeros(0)
        return hv_total, hv_without_each_tensor

    def _compute_global_hv_contributions(
        self,
        group_vectors: torch.Tensor,
        ref_point: torch.Tensor,
        pareto_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute HV(P ∪ {v_i}, r) - HV(P, r) for each ``v_i`` in a group.

        This method is used when the global Pareto cache is enabled. The
        ``pareto_vectors`` tensor is a snapshot of the global Pareto front for
        this batch and is *not* modified inside this method.

        Args:
            group_vectors: Tensor of shape (group_size, dim) with group scores.
            ref_point: Reference point tensor of shape (dim,).
            pareto_vectors: Tensor of shape (K, dim) with cached Pareto points.

        Returns:
            A tuple ``(hv_pareto, contribs)`` where:
            - hv_pareto: scalar tensor, ``HV(P, ref_point)``.
            - contribs: tensor of shape (group_size,), where the i-th element is
              ``HV(P ∪ {v_i}, ref_point) - HV(P, ref_point)``.
        """
        group_size, dim = group_vectors.shape
        if group_size == 0 or dim == 0:
            return group_vectors.new_tensor(0.0), group_vectors.new_zeros(group_size)

        if pareto_vectors.numel() == 0:
            hv_pareto = group_vectors.new_tensor(0.0)
        else:
            hv_pareto = self._hv_recursive_slicing(pareto_vectors, ref_point)

        contribs = group_vectors.new_zeros(group_size, dtype=torch.float32)
        for i in range(group_size):
            vi = group_vectors[i : i + 1]
            if pareto_vectors.numel() == 0:
                union_points = vi
            else:
                union_points = torch.cat([pareto_vectors, vi], dim=0)
            hv_with_i = self._hv_recursive_slicing(union_points, ref_point)
            contribs[i] = (hv_with_i - hv_pareto).to(torch.float32)

        return hv_pareto.to(torch.float32), contribs

    def _hv_recursive_slicing(
        self,
        points: torch.Tensor,
        ref_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hypervolume using recursive slicing algorithm."""
        hv: float = self._hv_recursive_slicing_helper(
            points.tolist(),
            ref_point.tolist(),
        )
        return torch.tensor(hv, dtype=ref_point.dtype, device=ref_point.device).clamp(min=0.0)

    def _hv_recursive_slicing_helper(
        self,
        points: List[tuple],
        ref_point: tuple,
    ) -> float:
        """Recursive slicing algorithm (list-based implementation).

        Adapted from Fonseca et al. (2006).
        """
        if not points:
            return 0.0

        # 1-D case
        if len(ref_point) == 1:
            return max(p[0] for p in points) - ref_point[0]

        points = sorted(points, key=lambda p: p[0])

        hv = 0.0
        ref0 = ref_point[0]  # moving slice position
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

    # ------------------------------------------------------------------
    # Global Pareto cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dominates(a: list[float], b: list[float], eps: float) -> bool:
        """Return True if ``a`` Pareto-dominates ``b`` under maximization.

        ``a`` dominates ``b`` if it is no worse in every coordinate and
        strictly better in at least one, up to tolerance ``eps``.
        """

        assert len(a) == len(b)
        better_in_any = False
        for x, y in zip(a, b):
            if x + eps < y:  # a is strictly worse in this coordinate
                return False
            if x > y + eps:
                better_in_any = True
        return better_in_any

    @classmethod
    def _filter_nondominated(cls, points: list[list[float]], eps: float) -> list[list[float]]:
        """Quadratic-time non-dominated filtering for arbitrary dimension.

        This is used to maintain an approximate global Pareto front. Complexity
        is acceptable for the small cache sizes used here (e.g. K <= 1024).
        """

        n = len(points)
        if n == 0:
            return []

        dominated = [False] * n
        for i in range(n):
            if dominated[i]:
                continue
            pi = points[i]
            for j in range(n):
                if i == j or dominated[i]:
                    continue
                pj = points[j]
                if cls._dominates(pj, pi, eps):
                    dominated[i] = True
                    break

        result: list[list[float]] = []
        for i, p in enumerate(points):
            if not dominated[i]:
                result.append(p)
        return result

    def _update_pareto_cache(self, new_points: list[list[float]]) -> None:
        """Update the global Pareto cache with new objective vectors.

        The cache stores only objective vectors (no metadata) and maintains a
        bounded set of non-dominated points under maximization. When the cache
        exceeds ``pareto_cache_max_size``, we evict the oldest points (FIFO)
        after non-dominated filtering, keeping the most recent points.
        """

        if not new_points:
            return

        if self.pareto_cache_max_size <= 0:
            # Effectively disable the cache while keeping the code paths simple.
            self._pareto_cache = []
            self._pareto_dim = None
            return

        # Determine and validate dimensionality.
        first_dim = len(new_points[0])
        for idx, p in enumerate(new_points):
            if len(p) != first_dim:
                raise ValueError(
                    f"[Amo][HV] new_points[{idx}] has dimension {len(p)}, expected {first_dim}."
                )

        if self._pareto_dim is None:
            self._pareto_dim = first_dim
        elif self._pareto_dim != first_dim:
            raise ValueError(
                f"[Amo][HV] Pareto cache dimension {self._pareto_dim} does not match new points dimension {first_dim}."
            )

        # Sanity-check existing cache.
        for idx, p in enumerate(self._pareto_cache):
            if len(p) != self._pareto_dim:
                raise ValueError(
                    f"[Amo][HV] Cached point at index {idx} has dimension {len(p)}, expected {self._pareto_dim}."
                )

        eps = float(self.pareto_cache_eps)

        # 1) Filter non-dominated points among the new candidates themselves.
        new_nd = self._filter_nondominated(list(new_points), eps)

        # 2) Drop existing cache points dominated by any new non-dominated point.
        remaining_cache: list[list[float]] = []
        for old in self._pareto_cache:
            if any(self._dominates(n, old, eps) for n in new_nd):
                continue
            remaining_cache.append(old)

        # 3) Drop new points that are dominated by any remaining cache point.
        filtered_new: list[list[float]] = []
        for cand in new_nd:
            if any(self._dominates(old, cand, eps) for old in remaining_cache):
                continue
            filtered_new.append(cand)

        # 4) Append new points and enforce FIFO capacity.
        merged = remaining_cache + filtered_new
        if len(merged) > self.pareto_cache_max_size:
            merged = merged[-self.pareto_cache_max_size :]

        self._pareto_cache = merged
