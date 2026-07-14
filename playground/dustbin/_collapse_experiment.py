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
"""ONE-OFF collapse-control experiment (P0 diagnosis).

⚠️  THROWAWAY / DIAGNOSTIC ONLY — NOT part of the benchmark suite.
    After it produces its numbers this file is moved to playground/dustbin/.
    The `HVPOCollapsedSurvival` here deliberately REPRODUCES THE BUG in the
    LLM-side reward manager. Never import it into real code.

Goal
----
Prove the causal chain behind why LLM-side HVPO collapsed while the moo_suite
HVPO wins:

    inflated global Pareto front  →  exclusive-HV contribution ≈ 0 for ~all
    group members  →  signal collapses to noise  →  HVPO degrades toward
    vanilla / random.

The shipped `HVPOSurvival` (in moo_suite.py) computes the greedy exclusive-HV
selection **within the current population only**, with a per-generation bound.
That is the "aligned / P0-correct" form and it wins 10/22 problems.

The LLM reward manager instead scores each rollout against a persistent global
Pareto cache that inflates toward the reward-model ceiling (max_size=1024),
so within a 4-sample rollout group almost nothing lies outside that front and
ΔHV = 0 for ~94% of samples (measured in the training logs).

This script adds a `HVPOCollapsedSurvival` that injects exactly that failure
mode into the SAME GA harness: it keeps a growing archive of every point ever
seen and measures each candidate's exclusive-HV *against that inflated archive*
instead of against the current population. Everything else (problem, operators,
pop size, seed) is identical, so any degradation is attributable solely to the
inflated-front mechanism.

Usage
-----
    python playground/benchmarks/_collapse_experiment.py \
        --problems zdt1 zdt2 dtlz2 wfg4 --seeds 3
"""

import argparse
import os
import sys

import numpy as np

# Import the SHIPPED suite pieces without modifying them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moo_suite as suite  # noqa: E402
from moo_suite import (  # noqa: E402
    HVPOSurvival,
    SurvivalGA,
    VanillaSurvival,
    RandomSurvival,
    make_problem,
    normalized_indicators,
)
from pymoo.optimize import minimize  # noqa: E402


class HVPOCollapsedSurvival(HVPOSurvival):
    """HVPO selection sabotaged by an inflated, persistent global front.

    Reproduces the LLM reward-manager bug: exclusive-HV of each overflow-front
    member is measured against a growing archive of everything seen so far
    (capped like the real cache) rather than against the current population.
    As the archive inflates toward the objective-space frontier, ΔHV → 0 for
    almost every candidate and the trimming decision degenerates.
    """

    def __init__(self, exact_hv=False, cache_max_size=1024):
        super().__init__(exact_hv=exact_hv)
        self._archive = None          # (K, n_obj) minimization-space points
        self._cache_max_size = cache_max_size

    def _update_archive(self, F):
        F = np.atleast_2d(np.asarray(F, dtype=float))
        if self._archive is None:
            self._archive = F.copy()
        else:
            self._archive = np.vstack([self._archive, F])
        # FIFO cap, mirroring ParetoCache.max_size behaviour.
        if len(self._archive) > self._cache_max_size:
            self._archive = self._archive[-self._cache_max_size:]

    def _do(self, problem, pop, n_survive=None, **kwargs):
        F = pop.get("F")
        n = len(F)
        # Grow the inflated archive with the whole population every generation.
        self._update_archive(F)
        if n <= n_survive:
            return pop

        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        # Same fixed per-generation bound as the shipped version.
        span = F.max(axis=0) - F.min(axis=0)
        span = np.where(span > 1e-12, span, 1.0)
        bound = F.max(axis=0) + 0.1 * span

        # The inflated archive, mapped to maximization space with the SAME bound.
        archive_max = np.maximum(bound - self._archive, 0.0)

        from pymoo.indicators.hv import HV

        def hv_fn(rows):
            rows = np.atleast_2d(np.asarray(rows))
            if len(rows) == 0:
                return 0.0
            return float(HV(ref_point=np.zeros(rows.shape[1]))(-rows))

        selected = []
        for front in fronts:
            front = list(front)
            if len(selected) + len(front) <= n_survive:
                selected.extend(front)
                if len(selected) == n_survive:
                    break
                continue

            need = n_survive - len(selected)
            Ff = F[front]
            obj_max = np.maximum(bound - Ff, 0.0)

            # KEY DIFFERENCE: exclusive-HV of each member is measured against
            # the inflated ARCHIVE, not against the current population. When the
            # archive already dominates the objective region, every ΔHV ≈ 0 and
            # ties are broken arbitrarily (argmax of near-equal zeros) — exactly
            # the collapse seen in the LLM logs.
            hv_archive = hv_fn(archive_max)

            pool = list(range(len(front)))
            chosen = []
            for _ in range(need):
                best_j, best_gain = pool[0], -np.inf
                for j in pool:
                    cand = np.vstack([archive_max, obj_max[j]])
                    gain = hv_fn(cand) - hv_archive
                    if gain > best_gain:
                        best_gain, best_j = gain, j
                chosen.append(best_j)
                pool.remove(best_j)
            selected.extend([front[j] for j in chosen])
            break

        return pop[np.array(selected[:n_survive], dtype=int)]


def build_algo(method, pop_size):
    if method == "hvpo":                       # aligned / P0-correct
        return SurvivalGA(pop_size, HVPOSurvival())
    if method == "hvpo_collapsed":             # LLM-style inflated-front bug
        return SurvivalGA(pop_size, HVPOCollapsedSurvival())
    if method == "vanilla":                    # weighted-sum lower reference
        return SurvivalGA(pop_size, VanillaSurvival())
    if method == "random":                     # random lower bound
        return SurvivalGA(pop_size, RandomSurvival())
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="*", default=["zdt1", "zdt2", "dtlz2", "wfg4"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-gen", type=int, default=150)
    ap.add_argument("--pop-size", type=int, default=60)
    args = ap.parse_args()

    methods = ["hvpo", "hvpo_collapsed", "vanilla", "random"]

    # Resolve (name, n_obj, n_var) from the suite's own problem list.
    plist = {p[0]: p for p in suite.build_problem_list()}

    print("=" * 78)
    print(f"COLLAPSE CONTROL  |  problems={args.problems}  seeds={args.seeds} "
          f"n_gen={args.n_gen} pop={args.pop_size}")
    print("  hvpo            = aligned (exclusive-HV within current population)")
    print("  hvpo_collapsed  = LLM-style bug (exclusive-HV vs inflated global archive)")
    print("=" * 78)

    agg = {m: {"hv": [], "igd": []} for m in methods}
    for pname in args.problems:
        if pname not in plist:
            print(f"[skip] {pname}: not in suite problem list")
            continue
        _, n_obj, n_var = plist[pname]
        prob = make_problem(pname, n_obj, n_var)
        pf = np.atleast_2d(prob.pareto_front())

        print(f"\n### {pname}  (n_obj={n_obj})")
        for m in methods:
            hvs, igds = [], []
            for s in range(args.seeds):
                problem = make_problem(pname, n_obj, n_var)
                algo = build_algo(m, args.pop_size)
                res = minimize(problem, algo, ("n_gen", args.n_gen), seed=s, verbose=False)
                F = res.F
                if F is None or len(F) == 0:
                    hvs.append(0.0); igds.append(float("inf")); continue
                hv, igd = normalized_indicators(np.atleast_2d(F), pf)
                hvs.append(hv); igds.append(igd)
            hv_m, hv_s = float(np.mean(hvs)), float(np.std(hvs))
            igd_m = float(np.mean([x for x in igds if np.isfinite(x)] or [float("inf")]))
            agg[m]["hv"].append(hv_m); agg[m]["igd"].append(igd_m)
            print(f"  {m:16s} HV={hv_m:.4f}±{hv_s:.4f}  IGD={igd_m:.4f}")

    print("\n" + "=" * 78)
    print("SUMMARY (mean over problems)")
    for m in methods:
        print(f"  {m:16s} HV={np.mean(agg[m]['hv']):.4f}  IGD={np.mean(agg[m]['igd']):.4f}")
    print("=" * 78)
    print("Expected: hvpo >> hvpo_collapsed, and hvpo_collapsed sinks toward")
    print("vanilla/random — i.e. the inflated global front is what kills HVPO.")


if __name__ == "__main__":
    main()
