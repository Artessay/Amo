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

from typing import List

import torch


class HypervolumeCalculator:
    """Hypervolume calculator using recursive slicing algorithm.

    This class provides methods to compute hypervolume and hypervolume contributions
    for multi-objective optimization problems.

    Convention: all objectives are treated under **maximization** and the
    hypervolume is the volume of the region dominated by ``points`` and bounded
    below by ``ref_point``. Points must dominate (be >= in every coordinate)
    the reference point to contribute positive volume.
    """

    @classmethod
    def calculate_hypervolume(
        cls,
        points: torch.Tensor,
        ref_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hypervolume using recursive slicing algorithm.

        Dominated points are filtered out first. This is both a correctness
        safeguard (the slicing routine assumes a mutually non-dominated set is
        harmless but redundant points waste work) and a speed optimization for
        the small fronts used here.
        """
        pts = points.tolist()
        ref = ref_point.tolist()
        pts = cls._filter_nondominated(pts, ref)
        hv: float = cls._calculate_hypervolume(pts, ref)
        return torch.tensor(hv, dtype=ref_point.dtype, device=ref_point.device)

    @staticmethod
    def _filter_nondominated(points: List[list], ref: tuple) -> List[list]:
        """Keep only points that (a) dominate the reference point in every
        coordinate and (b) are not Pareto-dominated by another point.

        Maximization convention. ``a`` dominates ``b`` if ``a`` is >= ``b`` in
        every coordinate and strictly greater in at least one.
        """
        # Drop points that do not dominate the reference point; they add no
        # volume and can produce negative widths in the slicing routine.
        cand = [p for p in points if all(pi >= ri for pi, ri in zip(p, ref))]
        n = len(cand)
        if n <= 1:
            return cand

        keep = [True] * n
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(n):
                if i == j or not keep[j]:
                    continue
                # j dominates i ?
                ge = all(cj >= ci for cj, ci in zip(cand[j], cand[i]))
                gt = any(cj > ci for cj, ci in zip(cand[j], cand[i]))
                if ge and gt:
                    keep[i] = False
                    break
        return [cand[i] for i in range(n) if keep[i]]

    @classmethod
    def _calculate_hypervolume(
        cls,
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

        hv = cls._recursive_slicing_algorithm(points, ref_point)
        assert hv >= 0.0, "Hypervolume must be non-negative"

        return hv

    @classmethod
    def _recursive_slicing_algorithm(
        cls,
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
                hv += width * cls._recursive_slicing_algorithm(slice_pts, slice_ref)
                ref0 = p0[0]
            # Keep only points strictly beyond the present slice
            points = [p for p in points if p[0] > p0[0]]

        return hv

if __name__ == "__main__":
    print("=== Test Hypervolume Calculator ===")

    # Pareto Frontier are surounded by (3, 4), (4, 3)
    points = [(1, 2), (2, 1), (3, 4), (4, 3)]
    ref_point = (0, 0)
    hv = HypervolumeCalculator._calculate_hypervolume(
        HypervolumeCalculator._filter_nondominated([list(p) for p in points], ref_point),
        ref_point,
    )
    print(f"Hypervolume: {hv}")
    assert hv == 15.0

    # Tensor API, dominated points should be ignored.
    pts = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    ref = torch.tensor([0.0, 0.0])
    hv_t = HypervolumeCalculator.calculate_hypervolume(pts, ref)
    assert abs(hv_t.item() - 15.0) < 1e-6, hv_t.item()

    # Points below the reference point add zero volume.
    pts2 = torch.tensor([[3.0, 4.0], [4.0, 3.0], [-1.0, 100.0]])
    hv2 = HypervolumeCalculator.calculate_hypervolume(pts2, ref)
    assert abs(hv2.item() - 15.0) < 1e-6, hv2.item()

    print("=== Test Complete ===")

