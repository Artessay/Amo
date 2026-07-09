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
"""Offline controlled benchmark: HVPO reward vs. vanilla weighted-sum reward.

This is a *fast, CPU-only, no-GPU, no-server* experiment on the standard
multi-objective optimization (MOO) test problems ZDT1 and DTLZ2. It isolates the
effect of the **optimization objective** used to select survivors in an
otherwise identical ``(mu + lambda)`` search:

    * ``vanilla``  -- survivors = top-mu by equal-weighted sum of objectives
                      (what ``AmoVanillaRewardManager`` optimizes).
    * ``hvpo``     -- survivors = greedy maximum-hypervolume subset, i.e. each
                      pick maximizes the *exclusive* HV contribution given those
                      already kept (the SMS-EMOA principle, and exactly the
                      signal the fixed ``AmoHvpoRewardManager`` assigns).

Both schemes share *everything else*: initialization, mutation operator, pool
size, number of rounds and the random seed. The only difference is the survival
criterion. We then measure the quality of the final *population* with two
standard MOO indicators:

    * Hypervolume (HV, higher is better)  -- coverage of objective space.
    * Inverted Generational Distance (IGD, lower is better) -- distance to the
      true Pareto front.

Expected result: the weighted-sum reward drives the whole population toward a
single knee/corner of the front (high per-objective mean, but low HV and high
IGD), while the HVPO reward spreads the population across the front (high HV,
low IGD). This is precisely the multi-objective advantage HVPO is designed to
provide and that a scalar weighted-sum reward structurally cannot.

Usage:
    python playground/benchmarks/moo_benchmark.py
    python playground/benchmarks/moo_benchmark.py --problem dtlz2 --n-obj 3
"""

import argparse

import numpy as np
import torch

from verl.workers.reward_manager.amo_utils.hypervolume_calculator import HypervolumeCalculator
from verl.workers.reward_manager.amo_utils.hybrid_reward import HybridRewardModel


# ----------------------------------------------------------------------
# Test problems.  Internally objectives are *minimized* in f-space and then
# mapped to a *maximization* objective space  obj = C - f  (C = per-dim upper
# bound) so that the origin is a valid, dominated reference point for HV.
# ----------------------------------------------------------------------
class ZDT1:
    name = "zdt1"
    n_obj = 2

    def __init__(self, n_var: int = 6):
        self.n_var = n_var
        self.C = np.array([1.0, 1.0])  # f1,f2 in [0,1] on/near the front

    def f(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.mean(x[1:])
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2])

    def true_front_f(self, n: int = 200) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n)
        f2 = 1.0 - np.sqrt(f1)  # g == 1 on the front
        return np.stack([f1, f2], axis=1)


class DTLZ2:
    name = "dtlz2"

    def __init__(self, n_obj: int = 3, n_var: int = 8):
        self.n_obj = n_obj
        self.n_var = n_var
        self.C = np.full(n_obj, 2.0)  # f_i in [0, ~1.5]; C=2 keeps obj>0

    def f(self, x: np.ndarray) -> np.ndarray:
        m = self.n_obj
        xm = x[m - 1:]
        g = np.sum((xm - 0.5) ** 2)
        f = np.empty(m)
        for i in range(m):
            val = (1.0 + g)
            for j in range(m - 1 - i):
                val *= np.cos(x[j] * np.pi / 2.0)
            if i > 0:
                val *= np.sin(x[m - 1 - i] * np.pi / 2.0)
            f[i] = val
        return f

    def true_front_f(self, n: int = 200) -> np.ndarray:
        # Sample the unit sphere in the positive orthant (g == 0 on the front).
        rng = np.random.default_rng(0)
        pts = np.abs(rng.standard_normal((n, self.n_obj)))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts


def make_problem(name: str, n_obj: int):
    if name == "zdt1":
        return ZDT1()
    if name == "dtlz2":
        return DTLZ2(n_obj=n_obj)
    raise ValueError(f"Unknown problem: {name}")


# ----------------------------------------------------------------------
# MOO indicators
# ----------------------------------------------------------------------
def hypervolume(obj_max: np.ndarray) -> float:
    """HV of maximization objective vectors w.r.t. the origin."""
    if len(obj_max) == 0:
        return 0.0
    pts = torch.tensor(obj_max, dtype=torch.float64)
    ref = torch.zeros(obj_max.shape[1], dtype=torch.float64)
    return float(HypervolumeCalculator.calculate_hypervolume(pts, ref).item())


def igd(obj_max: np.ndarray, true_front_max: np.ndarray) -> float:
    """Inverted Generational Distance (lower is better)."""
    if len(obj_max) == 0:
        return float("inf")
    d = np.linalg.norm(true_front_max[:, None, :] - obj_max[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def nondominated_max(obj_max: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset under maximization."""
    n = len(obj_max)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if np.all(obj_max[j] >= obj_max[i]) and np.any(obj_max[j] > obj_max[i]):
                keep[i] = False
                break
    return obj_max[keep]


# ----------------------------------------------------------------------
# Survival-selection schemes.  Given a candidate pool of objective vectors
# (maximization), select `mu` survivors.  This is the ONLY thing that differs
# between "vanilla" and "hvpo".
# ----------------------------------------------------------------------
def select_weighted_sum(cand_obj: np.ndarray, mu: int) -> np.ndarray:
    """Vanilla scalarization: keep the top-`mu` by equal-weighted sum.

    This mirrors ``AmoVanillaRewardManager`` (reward = mean of objectives). It
    has no notion of spread, so it piles survivors onto the single knee of the
    front that maximizes the sum.
    """
    w = np.ones(cand_obj.shape[1]) / cand_obj.shape[1]
    scores = cand_obj @ w
    order = np.argsort(-scores)
    return order[:mu]


def _nondominated_sort(cand_obj: np.ndarray) -> list[list[int]]:
    """Partition indices into successive non-dominated fronts (maximization)."""
    n = len(cand_obj)
    dominated_by = [0] * n           # how many points dominate i
    dominates = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(cand_obj[j] >= cand_obj[i]) and np.any(cand_obj[j] > cand_obj[i]):
                dominated_by[i] += 1
                dominates[j].append(i)
    fronts, cur = [], [i for i in range(n) if dominated_by[i] == 0]
    seen_counts = dominated_by[:]
    while cur:
        fronts.append(cur)
        nxt = []
        for i in cur:
            for k in dominates[i]:
                seen_counts[k] -= 1
                if seen_counts[k] == 0:
                    nxt.append(k)
        cur = nxt
    return fronts


def select_greedy_hv(cand_obj: np.ndarray, mu: int) -> np.ndarray:
    """HVPO-style environmental selection (the SMS-EMOA principle).

    Survivors are filled front-by-front from a non-dominated sort; the single
    overflowing front is trimmed by keeping the members with the largest
    *exclusive* hypervolume contribution -- exactly the signal the fixed HVPO
    reward manager assigns. Non-dominated sorting keeps selection well-defined
    even before the population reaches the reference region, and HV trimming
    spreads the survivors across the whole front.
    """
    n, m = cand_obj.shape
    if n <= mu:
        return np.arange(n)

    ref = torch.zeros(m, dtype=torch.float64)
    cand_t = torch.tensor(cand_obj, dtype=torch.float64)

    selected: list[int] = []
    for front in _nondominated_sort(cand_obj):
        if len(selected) + len(front) <= mu:
            selected.extend(front)
            if len(selected) == mu:
                break
            continue

        # This front overflows: greedily keep the highest-HV-contribution subset.
        need = mu - len(selected)
        pool = list(front)
        chosen: list[int] = []
        for _ in range(need):
            if not chosen:
                base = torch.zeros((0, m), dtype=torch.float64)
                hv_base = 0.0
            else:
                base = cand_t[chosen]
                hv_base = float(HypervolumeCalculator.calculate_hypervolume(base, ref).item())
            best_idx, best_gain = pool[0], -float("inf")
            for j in pool:
                union = torch.cat([base, cand_t[j:j + 1]], dim=0)
                hv_j = float(HypervolumeCalculator.calculate_hypervolume(union, ref).item())
                gain = hv_j - hv_base
                if gain > best_gain:
                    best_gain, best_idx = gain, j
            chosen.append(best_idx)
            pool.remove(best_idx)
        selected.extend(chosen)
        break

    return np.array(selected[:mu], dtype=int)


# ----------------------------------------------------------------------
# (mu + lambda) evolutionary search.  Variation (mutation) is IDENTICAL across
# schemes; only `select_fn` differs.  This isolates the effect of using
# hypervolume vs. weighted-sum as the optimization objective.
# ----------------------------------------------------------------------
def run_search(problem, select_fn, *, mu=20, lam=40, rounds=40, sigma=0.12, seed=0):
    rng = np.random.default_rng(seed)
    n_var = problem.n_var
    C = problem.C

    def evaluate(x):
        f = problem.f(np.clip(x, 0.0, 1.0))
        return C - f  # maximization objective vector

    # Initialize parent population in decision space.
    pop_x = rng.uniform(0.0, 1.0, size=(mu, n_var))

    history = []
    for _ in range(rounds):
        # --- variation: create lambda offspring by mutating random parents ---
        parent_idx = rng.integers(0, mu, size=lam)
        offspring_x = np.clip(pop_x[parent_idx] + rng.normal(0.0, sigma, size=(lam, n_var)), 0.0, 1.0)

        # Candidate pool = current parents + offspring (mu + lambda).
        cand_x = np.concatenate([pop_x, offspring_x], axis=0)
        cand_obj = np.stack([evaluate(x) for x in cand_x])

        # --- survival selection (the only differing component) ---
        surv = select_fn(cand_obj, mu)
        pop_x = cand_x[surv]

        pop_obj = np.stack([evaluate(x) for x in pop_x])
        history.append(hypervolume(nondominated_max(pop_obj)))

    pop_obj = np.stack([evaluate(x) for x in pop_x])
    return pop_obj, history


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="zdt1", choices=["zdt1", "dtlz2"])
    ap.add_argument("--n-obj", type=int, default=3, help="only used for dtlz2")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=40)
    args = ap.parse_args()

    problem = make_problem(args.problem, args.n_obj)
    true_front_max = problem.C - problem.true_front_f(200)

    schemes = {"vanilla (weighted-sum)": select_weighted_sum, "hvpo (greedy HV)": select_greedy_hv}

    print("=" * 72)
    print(f"Problem: {problem.name}  (n_obj={problem.n_obj})   seeds={args.seeds}  rounds={args.rounds}")
    print("Same (mu+lambda) search & mutation; ONLY the survival criterion differs.")
    print("Metric on the POLICY POPULATION (what the search converges to):")
    print("  HV  = dominated hypervolume  (higher is better)")
    print("  IGD = distance to true Pareto front (lower is better)")
    print("=" * 72)

    results = {}
    for label, fn in schemes.items():
        hvs, igds, mean_objs = [], [], []
        for s in range(args.seeds):
            pop_obj, _ = run_search(problem, fn, rounds=args.rounds, seed=s)
            nd = nondominated_max(pop_obj)
            hvs.append(hypervolume(nd))
            igds.append(igd(nd, true_front_max))
            mean_objs.append(pop_obj.mean(axis=0))
        results[label] = (np.array(hvs), np.array(igds), np.mean(mean_objs, axis=0))
        print(f"\n[{label}]")
        print(f"  HV  : {np.mean(hvs):.4f} +/- {np.std(hvs):.4f}")
        print(f"  IGD : {np.mean(igds):.4f} +/- {np.std(igds):.4f}")
        print(f"  per-objective mean of population: {np.round(results[label][2], 3).tolist()}")

    hv_v = results["vanilla (weighted-sum)"][0].mean()
    hv_h = results["hvpo (greedy HV)"][0].mean()
    igd_v = results["vanilla (weighted-sum)"][1].mean()
    igd_h = results["hvpo (greedy HV)"][1].mean()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"  HV : hvpo {hv_h:.4f}  vs  vanilla {hv_v:.4f}   "
          f"-> HVPO is {'BETTER' if hv_h > hv_v else 'WORSE'} "
          f"({100 * (hv_h - hv_v) / max(hv_v, 1e-9):+.1f}%)")
    print(f"  IGD: hvpo {igd_h:.4f}  vs  vanilla {igd_v:.4f}   "
          f"-> HVPO is {'BETTER' if igd_h < igd_v else 'WORSE'} "
          f"({100 * (igd_h - igd_v) / max(igd_v, 1e-9):+.1f}%)")
    print("=" * 72)

    ok = (hv_h > hv_v) and (igd_h < igd_v)
    print("\nRESULT:", "HVPO outperforms vanilla on both HV and IGD." if ok
          else "HVPO did not dominate vanilla on both metrics (try more seeds/rounds).")
    return ok


if __name__ == "__main__":
    main()
