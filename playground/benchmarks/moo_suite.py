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
"""Multi-objective optimization (MOO) benchmark suite.

Compares the Amo project's HVPO selection (the fixed exclusive-hypervolume
machinery in ``verl.workers.reward_manager.amo_utils``) against the vanilla
weighted-sum selection and several standard MOO baselines on the classic
synthetic test suites ZDT / DTLZ / WFG (all with mathematically exact Pareto
fronts, so IGD is trustworthy).

Methods
-------
Custom-survival methods share an IDENTICAL genetic-algorithm harness (same SBX
crossover, PM mutation, tournament selection, population size and generation
budget); only the *survival / environmental-selection* rule differs -- this
isolates the effect of the selection objective, exactly as HVPO vs. vanilla does
inside the LLM reward manager:

    * hvpo      -- non-dominated sort + exclusive hypervolume trimming, computed
                   with the project's own ``HypervolumeCalculator`` /
                   ``HybridRewardModel`` (the code shipped in Amo).
    * vanilla   -- top-N by equal-weighted sum of objectives
                   (mirrors ``AmoVanillaRewardManager``).
    * chebyshev -- top-N by weighted Tchebycheff scalarization.
    * random    -- random survival (lower bound / sanity control).

Reference baselines use pymoo's own implementations:

    * nsga2, nsga3, moead, smsemoa.

Indicators
----------
For each problem we normalize objectives to [0, 1] using the true Pareto front's
ideal/nadir, then report:

    * HV  (hypervolume, ref point = 1.1 in every normalized dim; higher better).
    * IGD (inverted generational distance to the true front; lower better).

Usage
-----
    python playground/benchmarks/moo_suite.py                # full sweep
    python playground/benchmarks/moo_suite.py --smoke        # tiny/fast check
    python playground/benchmarks/moo_suite.py --problems zdt1 dtlz2 --seeds 3
    python playground/benchmarks/moo_suite.py --aggregate-only   # rebuild table
"""

import argparse
import json
import os
import warnings
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.survival import Survival
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

import torch

# The project's own hypervolume machinery -- this is what we are validating.
from verl.workers.reward_manager.amo_utils.hypervolume_calculator import HypervolumeCalculator


HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(WORKSPACE, "results", "moo_benchmark")
RUNS_DIR = os.path.join(OUT_DIR, "runs")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")


# ----------------------------------------------------------------------
# Problem suite.  Each entry: (name, n_obj, n_var).  These defaults match the
# common literature settings and all have exact Pareto fronts in pymoo.
# ----------------------------------------------------------------------
def build_problem_list():
    problems = []
    for i in range(1, 7):
        problems.append((f"zdt{i}", 2, None))          # pymoo default n_var
    for i in range(1, 8):
        problems.append((f"dtlz{i}", 3, 12))
    for i in range(1, 10):
        problems.append((f"wfg{i}", 3, 12))
    return problems


# Families requested but NOT available offline with a trustworthy true front.
UNAVAILABLE = {
    "UF1-UF10": "CEC-2009 competition set; not shipped in pymoo (would need a "
                "hand port + external true fronts -> unreliable IGD).",
    "LSMOP1-LSMOP3": "Large-scale MOO set; PlatEMO-only, not in pymoo.",
    "DTLZ8-DTLZ9": "Constrained DTLZ variants; not provided by pymoo.",
}

METHODS = ["hvpo", "vanilla", "chebyshev", "random", "nsga2", "nsga3", "moead", "smsemoa"]
CUSTOM_SURVIVAL_METHODS = {"hvpo", "vanilla", "chebyshev", "random"}


def make_problem(name, n_obj, n_var):
    if n_var is None:
        return get_problem(name)
    return get_problem(name, n_var=n_var, n_obj=n_obj)


# ----------------------------------------------------------------------
# Custom survival operators (minimization space, as used by pymoo).
# ----------------------------------------------------------------------
class VanillaSurvival(Survival):
    """Equal-weighted sum survival -- the ``AmoVanillaRewardManager`` analogue."""

    def __init__(self):
        super().__init__(filter_infeasible=True)

    def _do(self, problem, pop, n_survive=None, **kwargs):
        F = pop.get("F")
        w = np.ones(F.shape[1]) / F.shape[1]
        order = np.argsort(F @ w)  # minimization: smaller sum is better
        return pop[order[:n_survive]]


class ChebyshevSurvival(Survival):
    """Weighted Tchebycheff scalarization survival (covers non-convex fronts)."""

    def __init__(self):
        super().__init__(filter_infeasible=True)

    def _do(self, problem, pop, n_survive=None, **kwargs):
        F = pop.get("F")
        z = F.min(axis=0)  # dynamic ideal point
        w = np.ones(F.shape[1]) / F.shape[1]
        s = np.max(w * (F - z), axis=1)
        order = np.argsort(s)
        return pop[order[:n_survive]]


class RandomSurvival(Survival):
    """Random survival -- a lower-bound sanity control."""

    def __init__(self):
        super().__init__(filter_infeasible=True)

    def _do(self, problem, pop, n_survive=None, random_state=None, **kwargs):
        n = len(pop)
        if n <= n_survive:
            return pop
        rs = random_state if random_state is not None else np.random
        idx = rs.permutation(n)[:n_survive]
        return pop[idx]


class HVPOSurvival(Survival):
    """HVPO-style environmental selection (the shipped Amo selection logic).

    Non-dominated sort fills survivors front-by-front; the single overflowing
    front is trimmed by greedily keeping the members with the largest
    *exclusive* hypervolume contribution. This is exactly the criterion the
    Amo HVPO reward manager assigns (``HybridRewardModel`` /
    ``HypervolumeCalculator``).

    Note on the HV backend: the greedy exclusive-HV *algorithm* here is identical
    to the reward manager's, but the inner hypervolume value is evaluated with a
    C-accelerated exact HV (pymoo/moocore) so the method can run at the same
    population/budget as every baseline. The project's own pure-Python
    ``HypervolumeCalculator`` is verified to return the same values in
    ``tests/amo/test_hvpo.py`` (it is exact but O(n^2), intended for the small
    rollout groups seen during LLM training, not pop=100 selection). The
    ``--exact-hv`` flag switches back to the project's calculator for spot checks.
    """

    def __init__(self, exact_hv=False):
        super().__init__(filter_infeasible=True)
        self.nds = NonDominatedSorting()
        self.exact_hv = exact_hv

    def _hv_project(self, obj_max_rows):
        """Exact HV via the project's own calculator (maximization, ref=origin)."""
        if len(obj_max_rows) == 0:
            return 0.0
        pts = torch.tensor(np.asarray(obj_max_rows), dtype=torch.float64)
        ref = torch.zeros(pts.shape[1], dtype=torch.float64)
        return float(HypervolumeCalculator.calculate_hypervolume(pts, ref).item())

    def _do(self, problem, pop, n_survive=None, **kwargs):
        F = pop.get("F")
        n = len(F)
        if n <= n_survive:
            return pop

        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        # Fixed reference / bound for the whole generation (SMS-EMOA style): use
        # the population nadir plus a margin so it is dominated by every point.
        # Keeping it fixed across candidates makes the greedy exclusive-HV values
        # comparable (a per-front dynamic bound would shift the objective each
        # call and destabilize selection).
        span = F.max(axis=0) - F.min(axis=0)
        span = np.where(span > 1e-12, span, 1.0)
        bound = F.max(axis=0) + 0.1 * span   # dominated upper reference

        selected = []
        for front in fronts:
            front = list(front)
            if len(selected) + len(front) <= n_survive:
                selected.extend(front)
                if len(selected) == n_survive:
                    break
                continue

            # Overflowing front: greedy exclusive-HV subset selection.
            need = n_survive - len(selected)
            Ff = F[front]
            # Map minimization objectives to maximization vectors, origin as the
            # dominated reference point (all entries >= 0 by construction).
            obj_max = np.maximum(bound - Ff, 0.0)

            if self.exact_hv:
                hv_fn = self._hv_project
            else:
                # pymoo HV minimizes; feed -obj_max with reference point at 0 so
                # it returns the dominated volume of obj_max above the origin.
                def hv_fn(rows, _HV=HV):
                    rows = np.atleast_2d(np.asarray(rows))
                    if len(rows) == 0:
                        return 0.0
                    return float(_HV(ref_point=np.zeros(rows.shape[1]))(-rows))

            pool = list(range(len(front)))
            chosen = []
            for _ in range(need):
                base = obj_max[chosen] if chosen else np.zeros((0, F.shape[1]))
                hv_base = hv_fn(base) if len(base) else 0.0
                best_j, best_gain = pool[0], -np.inf
                for j in pool:
                    cand = np.vstack([base, obj_max[j]]) if len(base) else obj_max[j][None, :]
                    gain = hv_fn(cand) - hv_base
                    if gain > best_gain:
                        best_gain, best_j = gain, j
                chosen.append(best_j)
                pool.remove(best_j)
            selected.extend([front[j] for j in chosen])
            break

        return pop[np.array(selected[:n_survive], dtype=int)]


SURVIVAL_REGISTRY = {
    "hvpo": HVPOSurvival,
    "vanilla": VanillaSurvival,
    "chebyshev": ChebyshevSurvival,
    "random": RandomSurvival,
}


def _binary_tournament(pop, P, random_state=None, **kwargs):
    """Random binary tournament (selection pressure comes from survival)."""
    rs = random_state if random_state is not None else np.random
    n = P.shape[0]
    S = np.empty(n, dtype=int)
    for i in range(n):
        a, b = P[i, 0], P[i, 1]
        S[i] = a if rs.random() < 0.5 else b
    return S


class SurvivalGA(GeneticAlgorithm):
    """Shared GA harness; the only per-method difference is ``survival``."""

    def __init__(self, pop_size, survival, **kwargs):
        super().__init__(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            selection=TournamentSelection(func_comp=_binary_tournament),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            survival=survival,
            advance_after_initial_infill=True,
            **kwargs,
        )


def build_algorithm(method, n_obj, pop_size, exact_hv=False):
    if method in CUSTOM_SURVIVAL_METHODS:
        if method == "hvpo":
            return SurvivalGA(pop_size, HVPOSurvival(exact_hv=exact_hv))
        return SurvivalGA(pop_size, SURVIVAL_REGISTRY[method]())

    if method in ("nsga3", "moead"):
        n_part = 99 if n_obj == 2 else 12
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_part)
        if method == "nsga3":
            return NSGA3(ref_dirs=ref_dirs)
        return MOEAD(ref_dirs=ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7)
    if method == "nsga2":
        return NSGA2(pop_size=pop_size)
    if method == "smsemoa":
        return SMSEMOA(pop_size=pop_size)
    raise ValueError(f"Unknown method: {method}")


# ----------------------------------------------------------------------
# Normalized indicators
# ----------------------------------------------------------------------
def normalized_indicators(F, pf):
    """Return (HV, IGD) in the [0,1]-normalized objective space of the problem.

    Normalization uses the true Pareto front's ideal/nadir so numbers are
    comparable across problems with different objective scales (e.g. WFG).
    """
    z_min = pf.min(axis=0)
    z_max = pf.max(axis=0)
    span = np.where(z_max - z_min > 1e-12, z_max - z_min, 1.0)

    Fn = (F - z_min) / span
    pfn = (pf - z_min) / span

    ref_point = np.full(F.shape[1], 1.1)
    hv = HV(ref_point=ref_point)(Fn)
    igd = IGD(pfn)(Fn)
    return float(hv), float(igd)


# ----------------------------------------------------------------------
# Single run
# ----------------------------------------------------------------------
def run_one(problem_name, n_obj, n_var, method, seed, n_gen, pop_size, pf, exact_hv=False):
    problem = make_problem(problem_name, n_obj, n_var)
    algo = build_algorithm(method, n_obj, pop_size, exact_hv=exact_hv)
    res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False)

    F = res.F
    if F is None or len(F) == 0:
        return {"hv": 0.0, "igd": float("inf"), "n_sol": 0, "F": []}

    F = np.atleast_2d(F)
    hv, igd = normalized_indicators(F, pf)
    return {"hv": hv, "igd": igd, "n_sol": int(len(F)), "F": F.tolist()}


def run_key(problem_name, method, seed):
    return f"{problem_name}__{method}__seed{seed}"


# ----------------------------------------------------------------------
# Sweep + persistence
# ----------------------------------------------------------------------
def sweep(problems, methods, seeds, n_gen, pop_size, resume=True, exact_hv=False):
    os.makedirs(RUNS_DIR, exist_ok=True)
    pf_cache = {}
    total = len(problems) * len(methods) * len(seeds)
    done = 0
    skipped_problems = []

    for (pname, n_obj, n_var) in problems:
        # Fetch the exact true front once per problem.
        try:
            if pname not in pf_cache:
                prob = make_problem(pname, n_obj, n_var)
                pf = prob.pareto_front()
                if pf is None or len(pf) == 0:
                    raise ValueError("empty pareto front")
                pf_cache[pname] = np.atleast_2d(pf)
        except Exception as e:  # noqa: BLE001
            skipped_problems.append((pname, str(e)[:60]))
            done += len(methods) * len(seeds)
            print(f"[skip] {pname}: cannot obtain true PF ({e})")
            continue

        pf = pf_cache[pname]
        for method in methods:
            for seed in seeds:
                done += 1
                out_path = os.path.join(RUNS_DIR, run_key(pname, method, seed) + ".json")
                if resume and os.path.exists(out_path):
                    print(f"[{done}/{total}] cached  {pname:8s} {method:9s} seed{seed}")
                    continue
                try:
                    rec = run_one(pname, n_obj, n_var, method, seed, n_gen, pop_size, pf,
                                  exact_hv=exact_hv)
                    status = "ok"
                except Exception as e:  # noqa: BLE001
                    rec = {"hv": 0.0, "igd": float("inf"), "n_sol": 0, "F": [], "error": str(e)[:120]}
                    status = f"ERR:{type(e).__name__}"
                rec.update({
                    "problem": pname, "n_obj": n_obj, "method": method, "seed": seed,
                    "n_gen": n_gen, "pop_size": pop_size,
                })
                with open(out_path, "w") as f:
                    json.dump(rec, f)
                print(f"[{done}/{total}] {status:14s} {pname:8s} {method:9s} seed{seed}  "
                      f"HV={rec['hv']:.4f} IGD={rec['igd']:.4f}")

    return skipped_problems


# ----------------------------------------------------------------------
# Aggregation -> main_table.md
# ----------------------------------------------------------------------
def load_runs():
    runs = []
    if not os.path.isdir(RUNS_DIR):
        return runs
    for fn in os.listdir(RUNS_DIR):
        if fn.endswith(".json"):
            with open(os.path.join(RUNS_DIR, fn)) as f:
                runs.append(json.load(f))
    return runs


def aggregate(problems, methods):
    runs = load_runs()
    # index: (problem, method) -> list of records
    idx = {}
    for r in runs:
        idx.setdefault((r["problem"], r["method"]), []).append(r)

    def stat(pname, method, key, finite_only=False):
        recs = idx.get((pname, method), [])
        vals = [r[key] for r in recs]
        if finite_only:
            vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    prob_names = [p[0] for p in problems]

    lines = []
    lines.append("# Multi-Objective Optimization Benchmark — Main Results\n")
    lines.append(
        "HVPO here is the Amo project's own exclusive-hypervolume selection "
        "(`verl/workers/reward_manager/amo_utils`), plugged into a shared GA "
        "harness so that **hvpo / vanilla / chebyshev / random differ only in "
        "the survival rule**. `nsga2/nsga3/moead/smsemoa` are pymoo reference "
        "implementations. Indicators are computed in the true-Pareto-front "
        "normalized `[0,1]` objective space.\n")
    lines.append(f"- Problems: **{len([p for p in prob_names])}** "
                 "(ZDT1–6, DTLZ1–7, WFG1–9), each with an exact Pareto front.\n")
    lines.append("- Seeds averaged per cell; **bold** = best method on that problem.\n")

    # ---- HV table ----
    def render_table(metric, better):
        hdr = "| Problem | " + " | ".join(methods) + " |"
        sep = "|" + "---|" * (len(methods) + 1)
        rows = [f"\n## {metric.upper()} ({'higher' if better=='max' else 'lower'} is better)\n", hdr, sep]
        # aggregate win counts
        wins = {m: 0 for m in methods}
        for pname in prob_names:
            means = {}
            for m in methods:
                mu, sd = stat(pname, m, metric, finite_only=(metric == "igd"))
                means[m] = (mu, sd)
            valid = {m: v[0] for m, v in means.items() if v[0] is not None and np.isfinite(v[0])}
            best_m = None
            if valid:
                best_m = (max if better == "max" else min)(valid, key=valid.get)
                wins[best_m] += 1
            cells = []
            for m in methods:
                mu, sd = means[m]
                if mu is None:
                    cells.append("—")
                elif not np.isfinite(mu):
                    cells.append("inf")
                else:
                    s = f"{mu:.3f}±{sd:.3f}"
                    if m == best_m:
                        s = f"**{s}**"
                    cells.append(s)
            rows.append(f"| {pname} | " + " | ".join(cells) + " |")
        rows.append(f"| **# best** | " + " | ".join(f"**{wins[m]}**" for m in methods) + " |")
        return "\n".join(rows), wins

    hv_table, hv_wins = render_table("hv", "max")
    igd_table, igd_wins = render_table("igd", "min")
    lines.append(hv_table)
    lines.append(igd_table)

    # ---- summary ----
    lines.append("\n## Summary (number of problems where each method ranks best)\n")
    lines.append("| Method | HV wins | IGD wins |")
    lines.append("|---|---|---|")
    for m in methods:
        lines.append(f"| {m} | {hv_wins[m]} | {igd_wins[m]} |")

    lines.append("\n## Families not included (unavailable offline)\n")
    for fam, why in UNAVAILABLE.items():
        lines.append(f"- **{fam}**: {why}")

    os.makedirs(OUT_DIR, exist_ok=True)
    table_path = os.path.join(OUT_DIR, "main_table.md")
    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[aggregate] wrote {table_path}")
    print(f"[aggregate] HV wins : {hv_wins}")
    print(f"[aggregate] IGD wins: {igd_wins}")
    return table_path


# ----------------------------------------------------------------------
# Pareto-front comparison plots (2-objective problems only -> readable scatter)
# ----------------------------------------------------------------------
def make_plots(problems, methods, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    runs = {(r["problem"], r["method"], r["seed"]): r for r in load_runs()}

    made = []
    for (pname, n_obj, n_var) in problems:
        if n_obj != 2:
            continue
        try:
            pf = np.atleast_2d(make_problem(pname, n_obj, n_var).pareto_front())
        except Exception:  # noqa: BLE001
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(pf[:, 0], pf[:, 1], "-", color="black", lw=1.2, label="true PF", zorder=1)
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        for m, c in zip(methods, colors):
            r = runs.get((pname, m, seed))
            if r is None or not r.get("F"):
                continue
            F = np.atleast_2d(np.array(r["F"]))
            ax.scatter(F[:, 0], F[:, 1], s=14, color=c, alpha=0.7, label=m, zorder=2)
        ax.set_title(f"{pname}  (seed {seed})")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"{pname}_front.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        made.append(out)
    print(f"[plots] wrote {len(made)} front plots to {PLOTS_DIR}")
    return made


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="*", default=None,
                    help="subset of problem names (default: full suite)")
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-gen", type=int, default=200)
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run (2 problems, 2 seeds, few gens) for validation")
    ap.add_argument("--no-resume", action="store_true", help="recompute cached runs")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip running; just rebuild the table and plots from runs/")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--exact-hv", action="store_true",
                    help="use the project's pure-Python HypervolumeCalculator for "
                         "HVPO selection (exact but slow; for spot-checking only)")
    args = ap.parse_args()

    all_problems = build_problem_list()
    if args.problems:
        wanted = set(args.problems)
        problems = [p for p in all_problems if p[0] in wanted]
    else:
        problems = all_problems

    methods = args.methods
    seeds = list(range(args.seeds))
    n_gen, pop_size = args.n_gen, args.pop_size

    if args.smoke:
        problems = [p for p in all_problems if p[0] in ("zdt1", "dtlz2")]
        methods = METHODS
        seeds = [0, 1]
        n_gen, pop_size = 40, 40

    print("=" * 78)
    print(f"MOO benchmark suite  |  problems={len(problems)} methods={len(methods)} "
          f"seeds={len(seeds)} n_gen={n_gen} pop={pop_size}")
    print("=" * 78)

    if not args.aggregate_only:
        skipped = sweep(problems, methods, seeds, n_gen, pop_size,
                        resume=not args.no_resume, exact_hv=args.exact_hv)
        if skipped:
            print("\nSkipped problems (no true PF offline):")
            for name, why in skipped:
                print(f"  {name}: {why}")

    aggregate(problems, methods)
    if not args.no_plots:
        make_plots(problems, methods, seed=seeds[0])


if __name__ == "__main__":
    main()
