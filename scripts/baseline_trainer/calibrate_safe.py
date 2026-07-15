#!/usr/bin/env python
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
"""[Amo] Frozen calibration for the PKU-SafeRLHF multi-objective baselines.

The two Safe-RLHF reward-model objectives (``safe_helpfulness``,
``safe_harmlessness``) are *unbounded* reward-model logits (empirically ~ -4..+8),
unlike the MATH objectives which are already in ``[0, 1]``. Several baseline
credit rules are scale-sensitive and therefore need calibration constants that,
per the fair-comparison protocol, must be:

  * estimated on a *frozen* calibration split (never the test statistics), and
  * shared verbatim across every method and every model.

This script scores the dataset's *reference* responses (``response_0`` /
``response_1`` -- a natural spread of safe and unsafe answers) on a fixed
calibration split of the train set, using the already-running reward gRPC
servers (helpful -> :50051, harmless -> :50052 on GPU 0/1). It needs **no extra
GPU** of its own; the reward models do the GPU work on the servers.

It writes ``results/PKU-SafeRLHF/safe_calibration.json`` with, per objective
(in the fixed order [safe_helpfulness, safe_harmlessness]):

  * ``calib_lower`` / ``calib_upper`` : robust p1 / p99 bounds -> affine
    normalization r~ = (r - lo) / (hi - lo) for ``amo_scalarize`` (LS /
    Tchebycheff). Maps the working reward range to ~[0, 1].
  * ``ideal``          : the calibrated utopia point (all-ones after affine
    normalization) for Tchebycheff.
  * ``hv_reference``   : the raw lower bound, a valid dominated reference point
    for the group-level HV weighting in ``dynamic_hv``.
  * ``harmless_budget``: a raw-scale constraint budget d for the Lagrangian
    (Safe-RLHF) baseline -- the median harmlessness of the *safe* reference
    responses, i.e. "keep harmlessness at least as high as a typical safe
    answer".
  * raw percentile diagnostics for reproducibility.

Usage (reward servers must be up: `bash scripts/amo_exp/serve_rewards.sh safe`):
    python scripts/baseline_trainer/calibrate_safe.py [--n 512] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# Make `recipe` importable when run from the repo root.
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from recipe.amo_safe.reward_client import compute_reward_score  # noqa: E402

# Fixed objective order -- MUST match the reward-function list order used by the
# training scripts: safe_helpfulness first, safe_harmlessness second.
OBJECTIVES = ["safe_helpfulness", "safe_harmlessness"]
HELP_HOST = os.getenv("HELPFUL_TARGET_HOST", "localhost")
HELP_PORT = os.getenv("HELPFUL_TARGET_PORT", "50051")
HARM_HOST = os.getenv("HARMLESS_TARGET_HOST", "localhost")
HARM_PORT = os.getenv("HARMLESS_TARGET_PORT", "50052")


def _score_one(args):
    prompt, response = args
    h = compute_reward_score(prompt, response, HELP_HOST, HELP_PORT)
    s = compute_reward_score(prompt, response, HARM_HOST, HARM_PORT)
    return float(h), float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(WORKSPACE, "data/PKU-SafeRLHF/train.parquet"))
    ap.add_argument("--out", default=os.path.join(WORKSPACE, "results/PKU-SafeRLHF/safe_calibration.json"))
    ap.add_argument("--n", type=int, default=512, help="# calibration prompts (uses both reference responses each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16, help="concurrent gRPC scoring threads")
    ap.add_argument("--lower_pct", type=float, default=1.0)
    ap.add_argument("--upper_pct", type=float, default=99.0)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
    sub = df.iloc[idx]

    jobs = []
    is_safe = []  # aligned with jobs: whether this reference response is labelled safe
    for _, row in sub.iterrows():
        question = row["extra_info"]["question"]
        for k in (0, 1):
            resp = row[f"response_{k}"]
            if not isinstance(resp, str) or not resp.strip():
                continue
            jobs.append((question, resp))
            is_safe.append(bool(row[f"is_response_{k}_safe"]))

    print(f"[calibrate] scoring {len(jobs)} reference responses "
          f"from {len(sub)} prompts via reward servers ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(_score_one, jobs))

    arr = np.array(results, dtype=np.float64)  # (N, 2): [helpful, harmless]
    is_safe = np.array(is_safe, dtype=bool)

    lower = np.percentile(arr, args.lower_pct, axis=0)
    upper = np.percentile(arr, args.upper_pct, axis=0)

    # Lagrangian harmlessness budget: median harmlessness of the *safe* references
    # (objective index 1). "Keep harmlessness >= a typical safe answer."
    harm = arr[:, 1]
    harmless_budget = float(np.median(harm[is_safe])) if is_safe.any() else float(np.median(harm))

    calib = {
        "_meta": {
            "objectives": OBJECTIVES,
            "n_prompts": int(len(sub)),
            "n_responses": int(len(jobs)),
            "seed": int(args.seed),
            "lower_pct": args.lower_pct,
            "upper_pct": args.upper_pct,
            "source": "PKU-SafeRLHF train reference responses (response_0/response_1)",
            "note": "Frozen calibration. Reuse verbatim across every method and model.",
        },
        # affine bounds for amo_scalarize (ls / tchebycheff)
        "calib_lower": [float(lower[0]), float(lower[1])],
        "calib_upper": [float(upper[0]), float(upper[1])],
        # after affine normalization objectives live in ~[0,1] -> utopia all-ones
        "ideal": [1.0, 1.0],
        # raw dominated reference for group-level HV weighting (dynamic_hv)
        "hv_reference": [float(lower[0]), float(lower[1])],
        # raw-scale constraint budget for the Lagrangian (Safe-RLHF) baseline
        "harmless_budget": harmless_budget,
        # diagnostics
        "raw_percentiles": {
            OBJECTIVES[j]: {
                "p1": float(np.percentile(arr[:, j], 1)),
                "p5": float(np.percentile(arr[:, j], 5)),
                "p50": float(np.percentile(arr[:, j], 50)),
                "p95": float(np.percentile(arr[:, j], 95)),
                "p99": float(np.percentile(arr[:, j], 99)),
                "mean": float(arr[:, j].mean()),
                "min": float(arr[:, j].min()),
                "max": float(arr[:, j].max()),
            }
            for j in range(len(OBJECTIVES))
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(calib, f, indent=2)
    print(json.dumps(calib, indent=2))
    print(f"[calibrate] wrote {args.out}")


if __name__ == "__main__":
    main()
