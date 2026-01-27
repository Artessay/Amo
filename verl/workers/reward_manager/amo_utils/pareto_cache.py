import threading
from typing import ClassVar, Optional


class ParetoCache:
    """Thread-safe singleton class to manage a global Pareto front cache.

    This cache maintains a bounded set of non-dominated points under maximization,
    supporting operations for initialization, updating, querying, and maintenance.

    Notes on singleton behavior:
    - Only one instance of ParetoCache can exist in a process.
    - The first construction call initializes the singleton's configuration.
    - Subsequent construction calls return the same instance and ignore new parameters.

    The cache behavior is controlled via:
    - max_size: Maximum number of stored Pareto points
    - eps: Dominance tolerance for approximate Pareto front
    - strategy: Eviction strategy (currently only "fifo")
    """

    # Singleton storage and construction lock (class-level).
    _instance: ClassVar[Optional["ParetoCache"]] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, max_size: int, eps: float, strategy: str = "fifo") -> "ParetoCache":
        """Create or return the singleton instance (thread-safe).

        This uses double-checked locking to avoid unnecessary locking after
        the singleton is created.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_size: int, eps: float, strategy: str = "fifo") -> None:
        """Initialize the singleton instance exactly once.

        Only the first call performs real initialization. Later calls
        return immediately and do not override existing configuration.
        """
        # Guard against re-initialization when __init__ is called multiple times.
        if getattr(self, "_initialized", False):
            return

        self.max_size = max_size
        self.eps = eps
        self.strategy = strategy

        if self.strategy not in {"fifo"}:
            raise ValueError(f"Unsupported pareto_cache_strategy: {self.strategy}")

        # Internal cache state (instance-level lock for cache operations).
        self._cache: list[list[float]] = []
        self._lock = threading.Lock()
        self._dim: int | None = None

        self._initialized = True

        print(
            f"[Amo] Pareto cache initialized with max_size={max_size}, eps={eps}, strategy={strategy}"
        )

    @staticmethod
    def _dominates(a: list[float], b: list[float], eps: float) -> bool:
        """Return True if `a` Pareto-dominates `b` under maximization.

        `a` dominates `b` if it is no worse in every coordinate and
        strictly better in at least one, up to tolerance `eps`.
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

        return [p for i, p in enumerate(points) if not dominated[i]]

    def get_snapshot(self) -> list[list[float]]:
        """Get a read-only snapshot of the current Pareto cache.

        Returns:
            A copy of the current Pareto cache points.
        """
        with self._lock:
            return [p[:] for p in self._cache]

    def update(self, new_points: list[list[float]]) -> None:
        """Update the global Pareto cache with new objective vectors.

        The cache stores only objective vectors (no metadata) and maintains a
        bounded set of non-dominated points under maximization. When the cache
        exceeds `max_size`, we evict the oldest points (FIFO) after non-dominated
        filtering, keeping the most recent points.

        Args:
            new_points: List of new objective vectors to add to the cache.
        """
        if self.max_size <= 0:
            # Effectively disable the cache while keeping the code paths simple.
            return

        if not new_points:
            return

        if isinstance(new_points[0], (float, int)):
            new_points = [list(new_points)]

        # Determine and validate dimensionality.
        first_dim = len(new_points[0])
        for idx, p in enumerate(new_points):
            if len(p) != first_dim:
                raise ValueError(
                    f"new_points[{idx}] has dimension {len(p)}, expected {first_dim}."
                )

        with self._lock:
            if self._dim is None:
                self._dim = first_dim
            elif self._dim != first_dim:
                raise ValueError(
                    f"Pareto cache dimension {self._dim} does not match new points dimension {first_dim}."
                )

            # Sanity-check existing cache.
            for idx, p in enumerate(self._cache):
                if len(p) != self._dim:
                    raise ValueError(
                        f"Cached point at index {idx} has dimension {len(p)}, expected {self._dim}."
                    )

            eps = float(self.eps)

            # 1) Filter non-dominated points among the new candidates themselves.
            new_nd = self._filter_nondominated(list(new_points), eps)

            # 2) Drop existing cache points dominated by any new non-dominated point.
            remaining_cache: list[list[float]] = []
            for old in self._cache:
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
            if len(merged) > self.max_size:
                merged = merged[-self.max_size :]

            self._cache = merged

    def size(self) -> int:
        """Get the current size of the Pareto cache.

        Returns:
            Number of points in the Pareto cache.
        """
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Clear the Pareto cache.

        This removes all stored points and resets the cache dimension.
        """
        with self._lock:
            self._cache = []
            self._dim = None


if __name__ == "__main__":

    print("=== ParetoCache Usage Example ===\n")

    # The first call initializes the singleton.
    cache = ParetoCache(max_size=10, eps=0.01)

    # Any subsequent call returns the same instance (parameters are ignored).
    cache2 = ParetoCache(max_size=999, eps=123.0)
    assert cache is cache2
    # Check that the parameters are set correctly
    assert cache.max_size == 10
    assert cache.eps == 0.01
    assert cache.strategy == "fifo"

    print("1. Adding initial points:")
    initial_points = [
        [1.0, 2.0],
        [2.0, 1.0],
        [1.5, 1.5],
    ]
    cache.update(initial_points)
    snapshot = cache.get_snapshot()
    print(f"   Added: {initial_points}")
    print(f"   Pareto front: {snapshot}")
    print(f"   Size: {len(snapshot)}\n")
    assert len(snapshot) == 3

    print("2. Adding dominated points (should not change the front):")
    dominated_points = [
        [0.5, 0.5],
        [1.0, 1.0],
    ]
    cache.update(dominated_points)
    snapshot = cache.get_snapshot()
    print(f"   Added: {dominated_points}")
    print(f"   Pareto front: {snapshot}")
    print(f"   Size: {len(snapshot)}\n")
    assert len(snapshot) == 3

    print("3. Adding new non-dominated points:")
    new_nd_points = [
        [2.5, 1.5],
        [0.8, 2.2],
    ]
    cache.update(new_nd_points)
    snapshot = cache.get_snapshot()
    print(f"   Added: {new_nd_points}")
    print(f"   Pareto front: {snapshot}")
    print(f"   Size: {len(snapshot)}\n")
    assert len(snapshot) == 3

    print("4. Adding points that dominate existing ones:")
    dominating_points = [
        [3.0, 3.0],
    ]
    cache.update(dominating_points)
    snapshot = cache.get_snapshot()
    print(f"   Added: {dominating_points}")
    print(f"   Pareto front: {snapshot}")
    print(f"   Size: {len(snapshot)}\n")
    assert len(snapshot) == 1

    print("5. Testing FIFO eviction (max_size=10):")
    many_points = [[i, 10 - i] for i in range(0, 10)]
    cache.update(many_points)
    snapshot = cache.get_snapshot()
    print(f"   Added {len(many_points)} points")
    print(f"   Pareto front size: {len(snapshot)}")
    print(f"   Pareto front: {snapshot}\n")
    assert len(snapshot) == 10

    print("6. Clearing the cache:")
    cache.clear()
    snapshot = cache.get_snapshot()
    print(f"   Cache cleared. Size: {len(snapshot)}\n")
    assert len(snapshot) == 0

    print("7. Add a single point:")
    single_point = [0.5, 0.5]
    cache.update(single_point)
    snapshot = cache.get_snapshot()
    print(f"   Added: {single_point}")
    print(f"   Pareto front: {snapshot}")
    print(f"   Size: {len(snapshot)}\n")
    assert len(snapshot) == 1

    print("=== Test Complete ===")
