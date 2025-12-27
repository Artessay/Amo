from __future__ import annotations

"""Exact hypervolume computation using a WFG-style recursive slicing algorithm.

This module implements dominated hypervolume for **maximization** problems,
using only pure Python lists and sorting. It provides:

- ``hypervolume``: exact HV for any dimension ``m >= 1``.
- ``hv_without_each``: HV(S \ {i}) for each input point.
- ``hv_contributions``: per-point contribution ``HV(S) - HV(S \ {i})``.

For ``m == 1`` and ``m == 2`` we use simple closed-form/2D exact formulas.
For ``m >= 3`` we use a WFG-style recursive slicing procedure that sweeps
along the first coordinate and recursively computes the (m-1)-dimensional
hypervolume of each slice. At every recursion level we filter dominated
points to reduce work and avoid double-counting.

The implementation is designed for small sets (K <= ~200) and moderate
dimension (m <= ~8), which is typical in RL reward shaping.

The public API works on sequences of Python floats. Optional helper
functions are provided to interoperate with PyTorch tensors when PyTorch
is available.
"""

from typing import Iterable, List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch as _torch

Point = Sequence[float]
Points = Sequence[Sequence[float]]

__all__ = [
    "hypervolume",
    "hv_without_each",
    "hv_contributions",
    "hypervolume_torch",
    "hv_without_each_torch",
    "hv_contributions_torch",
]


# ---------------------------------------------------------------------------
# Public API (Python lists)
# ---------------------------------------------------------------------------


def hypervolume(points: Points, ref_point: Point, *, eps: float = 1e-12) -> float:
    """Compute the dominated hypervolume of a set of points (maximization).

    The hypervolume is the Lebesgue measure of the union of axis-aligned
    boxes ``[ref_point, p]`` for all points ``p`` in ``points``.

    This function assumes **maximization**: larger objective values are
    better, and the reference point should be component-wise no larger
    than the points of interest (typically a "worst" point).

    Args:
        points:
            Iterable of points, each a sequence of floats of length ``m``.
        ref_point:
            Reference point of length ``m``. The dominated region is defined
            between ``ref_point`` and the Pareto front formed by ``points``.
        eps:
            Numerical tolerance used when comparing floats. Differences with
            absolute value ``<= eps`` are treated as ties, and widths ``<= eps``
            are considered zero.

    Returns:
        Exact dominated hypervolume as a ``float``.
    """

    pts = _ensure_points(points)
    ref = _ensure_ref_point(ref_point)
    return _hypervolume_from_points(pts, ref, eps)


def hv_without_each(points: Points, ref_point: Point, *, eps: float = 1e-12) -> List[float]:
    """Compute ``HV(S \ {i})`` for each point in ``points``.

    The i-th element of the result is the hypervolume of the set of all
    points except ``points[i]``. The ordering matches the input order.

    Args:
        points: Sequence of points ``S``.
        ref_point: Reference point for the hypervolume.
        eps: Numerical tolerance forwarded to :func:`hypervolume`.

    Returns:
        List of length ``len(points)`` with ``HV(S \ {i})`` for each ``i``.
    """

    pts = _ensure_points(points)
    ref = _ensure_ref_point(ref_point)
    n = len(pts)
    if n == 0:
        return []

    result: List[float] = []
    for i in range(n):
        subset = pts[:i] + pts[i + 1 :]
        hv_sub = _hypervolume_from_points(subset, ref, eps)
        result.append(hv_sub)
    return result


def hv_contributions(points: Points, ref_point: Point, *, eps: float = 1e-12) -> List[float]:
    """Per-point hypervolume contributions ``HV(S) - HV(S \ {i})``.

    Args:
        points: Sequence of points ``S``.
        ref_point: Reference point for the hypervolume.
        eps: Numerical tolerance forwarded to :func:`hypervolume`.

    Returns:
        List of per-point contributions, same length and order as ``points``.
        Small negative values due to numerical noise are clamped to zero.
    """

    pts = _ensure_points(points)
    ref = _ensure_ref_point(ref_point)
    n = len(pts)
    if n == 0:
        return []

    hv_total = _hypervolume_from_points(pts, ref, eps)
    contributions: List[float] = []
    for i in range(n):
        subset = pts[:i] + pts[i + 1 :]
        hv_sub = _hypervolume_from_points(subset, ref, eps)
        contrib = hv_total - hv_sub
        # Clamp tiny negative values caused by floating point noise
        if contrib < 0.0 and abs(contrib) <= (abs(hv_total) + 1.0) * 1e-10:
            contrib = 0.0
        contributions.append(contrib)
    return contributions


# ---------------------------------------------------------------------------
# Optional PyTorch interop
# ---------------------------------------------------------------------------


def _require_torch() -> "_torch":  # pragma: no cover - simple wrapper
    """Import and return ``torch`` or raise a helpful error if unavailable."""

    try:
        import torch as _torch  # type: ignore[assignment]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyTorch is required for this function but is not installed."
        ) from exc
    return _torch


def hypervolume_torch(
    points: "_torch.Tensor",
    ref_point: "_torch.Tensor",
    *,
    eps: float = 1e-12,
) -> "_torch.Tensor":
    """Hypervolume with PyTorch tensors.

    This is a thin wrapper around :func:`hypervolume` that converts
    tensors to Python lists and back to a scalar tensor.

    Args:
        points: Tensor of shape ``(n, m)``.
        ref_point: Tensor of shape ``(m,)``.
        eps: Numerical tolerance.

    Returns:
        A scalar tensor with the hypervolume value on ``points.device``.
    """

    torch = _require_torch()
    hv = hypervolume(points.tolist(), ref_point.tolist(), eps=eps)
    return torch.tensor(hv, dtype=points.dtype, device=points.device)


def hv_without_each_torch(
    points: "_torch.Tensor",
    ref_point: "_torch.Tensor",
    *,
    eps: float = 1e-12,
) -> "_torch.Tensor":
    """Tensor version of :func:`hv_without_each`.

    Args:
        points: Tensor of shape ``(n, m)``.
        ref_point: Tensor of shape ``(m,)``.
        eps: Numerical tolerance.

    Returns:
        1D tensor of shape ``(n,)`` containing ``HV(S \ {i})``.
    """

    torch = _require_torch()
    hv_list = hv_without_each(points.tolist(), ref_point.tolist(), eps=eps)
    return torch.tensor(hv_list, dtype=points.dtype, device=points.device)


def hv_contributions_torch(
    points: "_torch.Tensor",
    ref_point: "_torch.Tensor",
    *,
    eps: float = 1e-12,
) -> "_torch.Tensor":
    """Tensor version of :func:`hv_contributions`.

    Args:
        points: Tensor of shape ``(n, m)``.
        ref_point: Tensor of shape ``(m,)``.
        eps: Numerical tolerance.

    Returns:
        1D tensor of shape ``(n,)`` with per-point contributions.
    """

    torch = _require_torch()
    contrib_list = hv_contributions(points.tolist(), ref_point.tolist(), eps=eps)
    return torch.tensor(contrib_list, dtype=points.dtype, device=points.device)


# ---------------------------------------------------------------------------
# Internal helpers: validation & preprocessing
# ---------------------------------------------------------------------------


def _ensure_points(points: Points) -> List[List[float]]:
    """Convert arbitrary point container to ``List[List[float]]``.

    The function is conservative and does not modify the input.
    """

    pts: List[List[float]] = []
    for p in points:
        pts.append([float(x) for x in p])
    return pts


def _ensure_ref_point(ref_point: Point) -> List[float]:
    ref = [float(x) for x in ref_point]
    if not ref:
        raise ValueError("ref_point must have at least one dimension.")
    return ref


def _validate_dimensions(points: List[List[float]], ref: List[float]) -> None:
    m = len(ref)
    for idx, p in enumerate(points):
        if len(p) != m:
            raise ValueError(
                f"Point at index {idx} has dimension {len(p)}, "
                f"but expected {m}."
            )


def _hypervolume_from_points(
    points: List[List[float]],
    ref: List[float],
    eps: float,
) -> float:
    """Internal implementation shared by all public APIs.

    Assumes ``points`` and ``ref`` are already converted to lists of floats.
    """

    if not points:
        return 0.0

    _validate_dimensions(points, ref)

    # Filter out points that cannot contribute and deduplicate.
    preprocessed = _preprocess_points(points, ref, eps)
    if not preprocessed:
        return 0.0

    m = len(ref)
    if m == 1:
        return _hv_1d(preprocessed, ref, eps)
    if m == 2:
        return _hv_2d(preprocessed, ref, eps)

    # For m >= 3, use WFG-style recursive slicing on coordinates shifted so
    # that the reference point becomes the origin.
    shifted: List[List[float]] = [
        [p[d] - ref[d] for d in range(m)] for p in preprocessed
    ]
    # At the origin, dominance relations are unchanged by translation, but
    # we explicitly filter dominated points once to cut down the search.
    shifted_nd = _filter_nondominated(shifted, eps)
    return _wfg_recursive(shifted_nd, eps)


def _preprocess_points(
    points: List[List[float]],
    ref: List[float],
    eps: float,
) -> List[List[float]]:
    """Drop non-contributing points and approximate duplicates.

    A point is discarded if **any** coordinate is not strictly better than
    the reference (within ``eps``), because the box ``[ref, p]`` would have
    zero measure. Remaining points are deduplicated with an ``eps``-based
    equality check to keep the algorithm robust to ties and small noise.
    """

    m = len(ref)

    # First, filter out points that cannot generate positive volume.
    filtered: List[List[float]] = []
    for p in points:
        if any(p[d] <= ref[d] + eps for d in range(m)):
            continue
        filtered.append(p)

    if not filtered:
        return []

    # Then, remove approximate duplicates for robustness.
    unique: List[List[float]] = []
    for p in filtered:
        is_dup = False
        for q in unique:
            if all(abs(a - b) <= eps for a, b in zip(p, q)):
                is_dup = True
                break
        if not is_dup:
            unique.append(p)

    return unique


# ---------------------------------------------------------------------------
# Basic exact formulas for 1D and 2D
# ---------------------------------------------------------------------------


def _hv_1d(points: List[List[float]], ref: List[float], eps: float) -> float:
    """Exact 1D hypervolume for maximization.

    HV is simply the length from ``ref[0]`` to the maximum coordinate.
    """

    if not points:
        return 0.0
    max_val = max(p[0] for p in points)
    width = max_val - ref[0]
    if width <= eps:
        return 0.0
    return width


def _filter_nondominated_2d(points: List[List[float]], eps: float) -> List[List[float]]:
    """Filter dominated points in 2D for maximization.

    Returns a list of non-dominated points. Ties within ``eps`` are
    handled robustly.
    """

    if not points:
        return []

    # Sort by x ascending.
    pts = sorted(points, key=lambda p: p[0])

    # Scan from right to left, keeping points with strictly increasing y.
    nondominated: List[List[float]] = []
    max_y = float("-inf")
    for p in reversed(pts):
        y = p[1]
        if y > max_y + eps:
            nondominated.append(p)
            max_y = y
    nondominated.reverse()
    return nondominated


def _hv_2d(points: List[List[float]], ref: List[float], eps: float) -> float:
    """Exact 2D hypervolume for maximization.

    The algorithm first filters dominated points, then sorts the remaining
    non-dominated points by the first coordinate and applies the formula

    .. math::

        HV = \sum_k (x_k - x_{k-1}) * (y_k - r_y),  \quad x_0 = r_x.

    This matches the standard 2D exact formula used in many MOEA tools.
    """

    if not points:
        return 0.0

    pts = _filter_nondominated_2d(points, eps)
    if not pts:
        return 0.0

    pts = sorted(pts, key=lambda p: p[0])

    hv = 0.0
    prev_x = ref[0]
    r_y = ref[1]

    for x, y in pts:
        dx = x - prev_x
        if dx <= eps:
            # No positive width in this slice.
            prev_x = max(prev_x, x)
            continue
        dy = y - r_y
        if dy > eps:
            hv += dx * dy
        prev_x = x

    if hv <= 0.0:
        return 0.0
    return hv


# ---------------------------------------------------------------------------
# WFG-style recursive slicing for m >= 3
# ---------------------------------------------------------------------------


def _dominates(a: List[float], b: List[float], eps: float) -> bool:
    """Return True if ``a`` dominates ``b`` under maximization.

    ``a`` dominates ``b`` if ``a`` is no worse in every coordinate and
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


def _filter_nondominated(points: List[List[float]], eps: float) -> List[List[float]]:
    """Quadratic-time non-dominated filtering for arbitrary dimension.

    This is used inside the WFG recursion to cut down the number of points
    in each slice. Complexity is acceptable for the small K typical in
    RL reward shaping.
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
            if _dominates(pj, pi, eps):
                dominated[i] = True
                break

    result: List[List[float]] = []
    for i, p in enumerate(points):
        if not dominated[i]:
            result.append(p)
    return result


def _wfg_recursive(points: List[List[float]], eps: float) -> float:
    """WFG-style recursive hypervolume computation.

    ``points`` are assumed to be in **maximization** form with the
    reference point at the origin (i.e., coordinates are offsets
    ``p[d] - ref[d]`` and thus non-negative).

    The algorithm sweeps along the first coordinate, forming slices with
    positive width and recursively computing the hypervolume of the
    projected points in the remaining dimensions. At each recursion level
    we filter dominated points to avoid redundant work.
    """

    if not points:
        return 0.0

    dim = len(points[0])
    if dim == 1:
        # Origin-based 1D HV is just the maximum coordinate.
        max_val = max(p[0] for p in points)
        return max(max_val, 0.0)

    # Sort by the first coordinate (ascending).
    pts = sorted(points, key=lambda p: p[0])

    hv = 0.0
    ref0 = 0.0  # current slice origin along the first dimension

    while pts:
        x0 = pts[0][0]
        width = x0 - ref0

        if width > eps:
            # Project remaining points to the subspace of dimensions 2..dim.
            slice_points = [p[1:] for p in pts]
            # Filter dominated points in the subspace to avoid double-counting.
            slice_points = _filter_nondominated(slice_points, eps)
            slice_hv = _wfg_recursive(slice_points, eps)
            hv += width * slice_hv
            ref0 = x0

        # Remove all points whose first coordinate is not strictly larger
        # than the current boundary (within tolerance). They do not
        # participate in subsequent slices.
        current_x = x0
        pts = [p for p in pts if p[0] > current_x + eps]

    return hv
