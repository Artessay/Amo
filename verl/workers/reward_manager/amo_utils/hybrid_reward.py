
import torch
from verl.workers.reward_manager.amo_utils.hypervolume_calculator import HypervolumeCalculator


class HybridRewardModel:
    """Hybrid multi-objective reward based on hypervolume (HV) contribution.

    A trajectory that pushes the Pareto front outward is rewarded with the
    *volume* it adds; a dominated trajectory (that adds no volume) is instead
    given a small negative signal proportional to how far it is from the front,
    so the policy still receives gradient toward the frontier.

    Two reward modes are provided:

    * :meth:`compute_hybrid_reward` -- reward of a *single* point against a fixed
      reference front ``pareto_vectors`` (``ΔHV(v_i) = HV(P ∪ {v_i}) − HV(P)``).
      Useful when group members should be scored independently.

    * :meth:`compute_group_hybrid_rewards` -- **exclusive** HV contribution of
      each member *within its rollout group*::

          reward_i = HV(P ∪ G) − HV(P ∪ (G \\ {v_i})).

      This is the geometrically correct multi-objective credit assignment: two
      rollouts that land in the *same* new region of objective space no longer
      both collect the full volume of that region (removing one still leaves the
      other covering it), so the group is rewarded for **spreading** across the
      front rather than piling onto a single spot. This diversity pressure is
      exactly what a plain weighted-sum reward lacks.
    """

    # ------------------------------------------------------------------
    # Group-wise exclusive contribution (preferred)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_group_hybrid_rewards(
        group_vectors: torch.Tensor,   # (G, dim)
        pareto_vectors: torch.Tensor,  # (K, dim)
        ref_point: torch.Tensor,       # (dim,)
        distance_metric: str = "chebyshev",
    ) -> torch.Tensor:
        """Exclusive HV contribution (with distance fallback) for a group.

        Args:
            group_vectors: Objective vectors of the rollout group, shape (G, dim).
            pareto_vectors: Cached global Pareto front, shape (K, dim). May be
                empty (K == 0).
            ref_point: Reference point, shape (dim,).
            distance_metric: Fallback metric for dominated points
                ("chebyshev", "euclidean" or "none").

        Returns:
            Tensor of shape (G,) with a reward per group member.
        """
        group_size = group_vectors.shape[0]
        if group_size == 0:
            return group_vectors.new_zeros((0,))

        if pareto_vectors is None or pareto_vectors.numel() == 0:
            base = group_vectors.new_zeros((0, group_vectors.shape[1]))
        else:
            base = pareto_vectors

        # HV of the full union P ∪ G (computed once).
        union_all = torch.cat([base, group_vectors], dim=0)
        hv_union_all = HypervolumeCalculator.calculate_hypervolume(union_all, ref_point)

        # Reference front for the distance fallback = P ∪ G itself, so a
        # dominated member is measured against everything else it competes with.
        rewards = []
        for i in range(group_size):
            # Union without member i.
            if group_size == 1:
                others = base
            else:
                mask = torch.ones(group_size, dtype=torch.bool, device=group_vectors.device)
                mask[i] = False
                others = torch.cat([base, group_vectors[mask]], dim=0)

            hv_without_i = HypervolumeCalculator.calculate_hypervolume(others, ref_point)
            contribution = (hv_union_all - hv_without_i).to(torch.float32)

            if contribution > 0.0:
                rewards.append(contribution)
            else:
                # Dominated within P ∪ G: fall back to a distance penalty toward
                # the union front (excluding the point itself).
                point = group_vectors[i]
                ref_front = others
                distance = HybridRewardModel._distance_to_front(point, ref_front, distance_metric)
                rewards.append(-distance)

        trajectory_rewards = torch.stack(rewards)
        assert trajectory_rewards.shape == (group_size,), (
            f"[Amo][HV] Hybrid reward shape mismatch: {trajectory_rewards.shape}"
        )
        return trajectory_rewards

    # ------------------------------------------------------------------
    # Single-point contribution (legacy / independent scoring)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_hybrid_reward(
        point: torch.Tensor,  # (dim,)
        pareto_vectors: torch.Tensor,   # (K, dim)
        ref_point: torch.Tensor,     # (dim,)
        distance_metric: str = "chebyshev",
    ) -> torch.Tensor:
        """Hybrid reward for a single ``point`` against a fixed front.

        Returns ``ΔHV`` if the point expands the front, otherwise a negative
        distance-to-front penalty.
        """
        hv_contribution = HybridRewardModel.compute_hv_contribution_to_pareto(point, pareto_vectors, ref_point)

        if hv_contribution > 0.0:
            return hv_contribution

        distance = HybridRewardModel._distance_to_front(point, pareto_vectors, distance_metric)
        return -distance

    @staticmethod
    def _distance_to_front(
        point: torch.Tensor,
        pareto_vectors: torch.Tensor,
        distance_metric: str,
    ) -> torch.Tensor:
        if distance_metric == "chebyshev":
            return HybridRewardModel.compute_chebyshev_distance_to_pareto(point, pareto_vectors)
        if distance_metric == "euclidean":
            return HybridRewardModel.compute_euclidean_distance_to_pareto(point, pareto_vectors)
        if distance_metric == "none":
            return point.new_tensor(0.0)
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    @staticmethod
    def compute_hv_contribution_to_pareto(
        point: torch.Tensor,  # (dim,)
        pareto_vectors: torch.Tensor,   # (K, dim)
        ref_point: torch.Tensor,     # (dim,)
    ) -> torch.Tensor:
        """Compute HV(P ∪ {v_i}, r) - HV(P, r) for point ``v_i``.

        The ``pareto_vectors`` tensor is a snapshot of the global Pareto front
        for this batch and is *not* modified inside this method.
        """
        if pareto_vectors is None or pareto_vectors.numel() == 0:
            hv_pareto = point.new_tensor(0.0)
            union_points = point.unsqueeze(0)
        else:
            hv_pareto = HypervolumeCalculator.calculate_hypervolume(pareto_vectors, ref_point)
            union_points = torch.cat([pareto_vectors, point.unsqueeze(0)], dim=0)

        hv_with_point = HypervolumeCalculator.calculate_hypervolume(union_points, ref_point)

        contribution = (hv_with_point - hv_pareto).to(torch.float32)
        return contribution

    @staticmethod
    def compute_chebyshev_distance_to_pareto(
        point: torch.Tensor,  # (dim,)
        pareto_vectors: torch.Tensor,  # (K, dim)
    ) -> torch.Tensor:
        """Chebyshev "improvement distance" from a dominated point to the front.

        For each front point, the improvement needed in its worst dimension is
        ``max_d (front_d - point_d)``; we take the minimum such gap over the
        front (the easiest front point to reach). Non-positive gaps are clamped
        to zero (the point already dominates that front point in every
        coordinate, so no improvement is required).
        """
        if pareto_vectors is None or pareto_vectors.numel() == 0:
            return point.new_tensor(0.0)

        gaps = pareto_vectors - point.unsqueeze(0)  # (K, dim)
        max_gaps = gaps.max(dim=1).values  # (K,)
        min_distance = max_gaps.min()
        # Robust clamp instead of a hard assert: with an approximate / tolerant
        # front the "distance" can be slightly negative, which is fine.
        return torch.clamp_min(min_distance, 0.0)

    @staticmethod
    def compute_euclidean_distance_to_pareto(
        point: torch.Tensor,  # (dim,)
        pareto_vectors: torch.Tensor,  # (K, dim)
    ) -> torch.Tensor:
        """Euclidean improvement distance from a dominated point to the front.

        Only the shortfall in each dimension (``front_d - point_d`` clamped at 0)
        contributes, so dimensions where the point already exceeds the front do
        not reduce the distance. Normalized by ``sqrt(dim)`` for comparability
        across objective counts.
        """
        if pareto_vectors is None or pareto_vectors.numel() == 0:
            return point.new_tensor(0.0)

        gaps = torch.clamp_min(pareto_vectors - point.unsqueeze(0), 0.0)  # (K, dim)
        distances = gaps.norm(p=2, dim=1)  # (K,)
        min_distance = distances.min()
        min_distance = torch.clamp_min(min_distance, 0.0)

        dim = point.size(0)
        norm_distance = min_distance / torch.sqrt(torch.tensor(dim, dtype=point.dtype, device=point.device))
        return norm_distance


if __name__ == "__main__":
    print("Testing hybrid reward model...")

    # Test case 1: Empty Pareto front -> full volume of the point.
    point1 = torch.tensor([2.0, 3.0])
    pareto_vectors1 = torch.zeros((0, 2))
    ref_point1 = torch.tensor([0.0, 0.0])
    reward1 = HybridRewardModel.compute_hybrid_reward(point1, pareto_vectors1, ref_point1)
    print(f"Test 1 - Empty Pareto front: reward = {reward1.item()}")
    assert abs(reward1.item() - 6.0) < 1e-6

    # Test case 2: Point improves Pareto front.
    point2 = torch.tensor([4.0, 5.0])
    pareto_vectors2 = torch.tensor([[2.0, 3.0], [5.0, 2.0]])
    ref_point2 = torch.tensor([0.0, 0.0])
    reward2 = HybridRewardModel.compute_hybrid_reward(point2, pareto_vectors2, ref_point2)
    print(f"Test 2 - Point improves Pareto front: reward = {reward2.item()}")
    assert reward2.item() > 0.0

    # Test case 3: Dominated point -> negative distance penalty.
    point3 = torch.tensor([1.0, 1.0])
    pareto_vectors3 = torch.tensor([[2.0, 3.0], [5.0, 2.0]])
    ref_point3 = torch.tensor([0.0, 0.0])
    reward3 = HybridRewardModel.compute_hybrid_reward(point3, pareto_vectors3, ref_point3)
    print(f"Test 3 - Dominated point: reward = {reward3.item()}")
    assert reward3.item() < 0.0

    # Test case 4: Group exclusive contribution rewards diversity.
    # Two identical points on a new region should NOT both get the full volume.
    ref = torch.tensor([0.0, 0.0])
    front = torch.zeros((0, 2))
    diverse = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    duplicate = torch.tensor([[3.0, 1.0], [3.0, 1.0]])
    r_div = HybridRewardModel.compute_group_hybrid_rewards(diverse, front, ref)
    r_dup = HybridRewardModel.compute_group_hybrid_rewards(duplicate, front, ref)
    print(f"Test 4 - diverse group rewards: {r_div.tolist()}")
    print(f"Test 4 - duplicate group rewards: {r_dup.tolist()}")
    # Duplicate members share the region: each exclusive contribution ~ 0.
    assert r_div.sum().item() > r_dup.sum().item()

    print("All tests completed!")
