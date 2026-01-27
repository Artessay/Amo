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

from typing import List, Tuple

import torch


class HypervolumeCalculator:
    """Hypervolume calculator using recursive slicing algorithm.

    This class provides methods to compute hypervolume and hypervolume contributions
    for multi-objective optimization problems.
    """

    def calculate_hypervolume(
        self,
        points: torch.Tensor,
        ref_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hypervolume using recursive slicing algorithm."""
        hv: float = self._calculate_hypervolume(
            points.tolist(),
            ref_point.tolist(),
        )
        return torch.tensor(hv, dtype=ref_point.dtype, device=ref_point.device)

    def _calculate_hypervolume(
        self,
        points: List[tuple],
        ref_point: tuple,
    ) -> float:
        """Compute hypervolume for a set of points.

        Args:
            points: List of points in the objective space.
            ref_point: Reference point for hypervolume calculation.

        Returns:
            Hypervolume value.
        """

        hv = self._recursive_slicing_algorithm(points, ref_point)
        assert hv >= 0.0, "Hypervolume must be non-negative"

        return hv

    def _recursive_slicing_algorithm(
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
                hv += width * self._recursive_slicing_algorithm(slice_pts, slice_ref)
                ref0 = p0[0]
            # Keep only points strictly beyond the present slice
            points = [p for p in points if p[0] > p0[0]]

        return hv

if __name__ == "__main__":
    hv_calculator = HypervolumeCalculator()

    print("=== Test Hypervolume Calculator ===")

    # Pareto Frontier are surounded by (3, 4), (4, 3)
    points = [(1, 2), (2, 1), (3, 4), (4, 3)]
    ref_point = (0, 0)
    hv = hv_calculator._calculate_hypervolume(points, ref_point)
    print(f"Hypervolume: {hv}")
    assert hv == 15.0
    
    print("=== Test Complete ===")

