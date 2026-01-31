
import torch
from verl.workers.reward_manager.amo_utils.hypervolume_calculator import HypervolumeCalculator

class HybridRewardModel:
    @staticmethod
    def compute_hybrid_reward(
        point: torch.Tensor,  # (dim,)
        pareto_vectors: torch.Tensor,   # (K, dim)
        ref_point: torch.Tensor,     # (dim,)

        distance_metric: str = "chebyshev",
    ) -> torch.Tensor:
        """Compute hybrid reward for a point.

        Args:
            point: Tensor of shape (dim,) with the point to evaluate.
            ref_point: Reference point tensor of shape (dim,).
            pareto_vectors: Tensor of shape (K, dim) with cached Pareto points.

        Returns:
            Hybrid reward value.
        """

        # Compute HV contribution to Pareto front
        hv_contribution = HybridRewardModel.compute_hv_contribution_to_pareto(point, pareto_vectors, ref_point)
        
        if hv_contribution > 0.0:
            # HV contribution is positive, use it as reward
            hybrid_reward = hv_contribution
        else:
            # HV contribution is negative or zero, use distance as reward
            if distance_metric == "chebyshev":
                distance = HybridRewardModel.compute_chebyshev_distance_to_pareto(point, pareto_vectors)
            elif distance_metric == "manhattan":
                distance = HybridRewardModel.compute_manhattan_distance_to_pareto(point, pareto_vectors)
            elif distance_metric == "euclidean":
                distance = HybridRewardModel.compute_euclidean_distance_to_pareto(point, pareto_vectors)
            elif distance_metric == "none":
                distance = point.new_tensor(0.0)
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")
            
            hybrid_reward = -distance
        
        return hybrid_reward
    
    @staticmethod
    def compute_hv_contribution_to_pareto(
        point: torch.Tensor,  # (dim,)
        pareto_vectors: torch.Tensor,   # (K, dim)
        ref_point: torch.Tensor,     # (dim,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute HV(P ∪ {v_i}, r) - HV(P, r) for point ``v_i``.

        This method is used when the global Pareto cache is enabled. The
        ``pareto_vectors`` tensor is a snapshot of the global Pareto front for
        this batch and is *not* modified inside this method.

        Args:
            point: Tensor of shape (dim,) with the point to evaluate.
            ref_point: Reference point tensor of shape (dim,).
            pareto_vectors: Tensor of shape (K, dim) with cached Pareto points.

        Returns:
            
        """
        if pareto_vectors.numel() == 0:
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
        """Compute the "improvement distance" from a point to the Pareto front.
        
        For dominated points, returns the amount of improvement needed to reach the front.
        Smaller return values indicate proximity to the front (better performance).
        """
        if pareto_vectors.numel() == 0:
            return torch.tensor(0.0, device=point.device)
        
        # Calculate the "dominance distance" to each Pareto point
        # For each dimension, calculate the amount of improvement needed (negative values mean already better)
        gaps = pareto_vectors - point.unsqueeze(0)  # (K, dim)
        
        # Use minimum positive distance (find the easiest point to catch up to)
        # For each Pareto point, calculate the maximum dimension that needs improvement
        max_gaps = gaps.max(dim=1).values  # (K,)
        
        # Find the Pareto point that is easiest to reach
        min_distance = max_gaps.min()
        assert min_distance >= 0.0 # all distance should be non-negative
        
        return min_distance
    
    @staticmethod
    def compute_manhattan_distance_to_pareto(
        point: torch.Tensor,  # (dim,)
        pareto_vectors: torch.Tensor,  # (K, dim)
    ) -> torch.Tensor:
        """Compute the "improvement distance" from a point to the Pareto front.
        
        For dominated points, returns the amount of improvement needed to reach the front.
        Smaller return values indicate proximity to the front (better performance).
        """
        if pareto_vectors.numel() == 0:
            return torch.tensor(0.0, device=point.device)
        
        # Calculate the "dominance distance" to each Pareto point
        # For each dimension, calculate the amount of improvement needed (negative values mean already better)
        gaps = pareto_vectors - point.unsqueeze(0)  # (K, dim)

        # Each element in gaps should be non-negative
        assert (gaps >= 0.0).all(), "All elements in gaps should be non-negative"
        
        # Use Manhattan distance to each Pareto point
        distances = gaps.abs().sum(dim=1)  # (K,), actually, abs is not necessary
        
        # Find the Pareto point that is easiest to reach
        min_distance = distances.min()
        assert min_distance >= 0.0 # all distance should be non-negative
        
        return min_distance

    @staticmethod
    def compute_euclidean_distance_to_pareto(
        point: torch.Tensor,  # (dim,)
        pareto_vectors: torch.Tensor,  # (K, dim)
    ) -> torch.Tensor:
        """Compute the "improvement distance" from a point to the Pareto front.
        
        For dominated points, returns the amount of improvement needed to reach the front.
        Smaller return values indicate proximity to the front (better performance).
        """
        if pareto_vectors.numel() == 0:
            return torch.tensor(0.0, device=point.device)
        
        # Calculate the "dominance distance" to each Pareto point
        # For each dimension, calculate the amount of improvement needed (negative values mean already better)
        gaps = pareto_vectors - point.unsqueeze(0)  # (K, dim)
        
        # Use Euclidean distance to each Pareto point
        distances = gaps.norm(p=2, dim=1)  # (K,)
        
        # Find the Pareto point that is easiest to reach
        min_distance = distances.min()
        assert min_distance >= 0.0 # all distance should be non-negative

        # Normalize the distance by the square root of the dimension
        # This ensures that the distance is comparable across different dimensions
        dim = point.size(0)
        norm_distance = min_distance / torch.sqrt(torch.tensor(dim, dtype=point.dtype, device=point.device))
        
        return norm_distance

if __name__ == "__main__":
    # Test compute_hybrid_reward function
    print("Testing hybrid reward model...")
    
    # Test case 1: Empty Pareto front
    point1 = torch.tensor([2.0, 3.0])
    pareto_vectors1 = torch.tensor([])
    ref_point1 = torch.tensor([0.0, 0.0])
    reward1 = HybridRewardModel.compute_hybrid_reward(point1, pareto_vectors1, ref_point1)
    print(f"Test 1 - Empty Pareto front: reward = {reward1.item()}")
    
    # Test case 2: Point improves Pareto front
    point2 = torch.tensor([4.0, 5.0])
    pareto_vectors2 = torch.tensor([[2.0, 3.0], [5.0, 2.0]])
    ref_point2 = torch.tensor([0.0, 0.0])
    reward2 = HybridRewardModel.compute_hybrid_reward(point2, pareto_vectors2, ref_point2)
    print(f"Test 2 - Point improves Pareto front: reward = {reward2.item()}")
    
    # Test case 3: Point does not improve Pareto front
    point3 = torch.tensor([1.0, 1.0])
    pareto_vectors3 = torch.tensor([[2.0, 3.0], [5.0, 2.0]])
    ref_point3 = torch.tensor([0.0, 0.0])
    reward3 = HybridRewardModel.compute_hybrid_reward(point3, pareto_vectors3, ref_point3)
    print(f"Test 3 - Point does not improve Pareto front: reward = {reward3.item()}")
    
    print("All tests completed!")
