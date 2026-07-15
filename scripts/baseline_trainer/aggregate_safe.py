#!/usr/bin/env python
# Copyright 2025 Rihong Qiu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""[Amo] Aggregate PKU-SafeRLHF baseline results into a comparison table.

Scans results/PKU-SafeRLHF/*.json (each produced by verl.trainer.amo_eval for a
single <MODEL_TAG>_<METHOD> experiment) and writes a Markdown table
results/PKU-SafeRLHF/baselines_table.md summarizing, per model x method:

  * safe_helpfulness (mean over test prompts),
  * safe_harmlessness (mean over test prompts),
  * hypervolume (dominated HV at ref = origin, as computed by amo_eval),
  * num_prompts.

Re-runnable at any time; safe to call after every completed cell. Methods and
models are printed in a fixed, meaningful order; missing cells are shown as "-".

Usage:
    python scripts/baseline_trainer/aggregate_safe.py [--results DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

DATA_SOURCE = "PKU-Alignment/PKU-SafeRLHF"

# Fixed display order.
MODELS = [
    ("qwen2.5-1.5b", "Qwen2.5-1.5B-Instruct"),
    ("qwen2.5-3b", "Qwen2.5-3B-Instruct"),
    ("llama3.2-3b", "Llama-3.2-3B-Instruct"),
]
# Reference online-RL methods first, then the 11 controlled baselines.
METHODS = [
    "grpo", "gdpo", "hvpo",
    "ls", "tchebycheff", "gdpo_weighted", "rvpo", "mgda", "gapo",
    "lagrangian", "fair_stable", "ctwa", "dynamic_hv", "nsga2", "smsemoa",
]
METHOD_LABEL = {
    "grpo": "GRPO (equal-weight)",
    "gdpo": "GDPO",
    "hvpo": "HVPO",
    "ls": "LS / MORLHF",
    "tchebycheff": "Tchebycheff",
    "gdpo_weighted": "GDPO (weighted)",
    "rvpo": "RVPO (soft-min)",
    "mgda": "MGDA",
    "gapo": "GAPO",
    "lagrangian": "Lagrangian (Safe-RLHF)",
    "fair_stable": "Fair-and-Stable",
    "ctwa": "CTWA",
    "dynamic_hv": "Dynamic-HV weighting",
    "nsga2": "NSGA-II-style credit",
    "smsemoa": "SMS-EMOA-style credit",
}


def load_all(results_dir: str) -> dict:
    """exp_name -> metrics dict (for DATA_SOURCE)."""
    out = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        exp = os.path.splitext(os.path.basename(path))[0]
        if exp == "safe_calibration":
            continue
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        if DATA_SOURCE in d:
            out[exp] = d[DATA_SOURCE]
    return out


def fmt(x, nd=3):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    ap.add_argument("--results", default=os.path.join(here, "results/PKU-SafeRLHF"))
    ap.add_argument("--out", default=os.path.join(here, "results/PKU-SafeRLHF/baselines_table.md"))
    args = ap.parse_args()

    data = load_all(args.results)

    lines = []
    lines.append("# PKU-SafeRLHF multi-objective baselines\n")
    lines.append(
        "Per-objective test means and dominated hypervolume (HV, ref = origin) "
        "for each model x method. Higher is better on all columns. Objectives are "
        "the Safe-RLHF reward-model scores `safe_helpfulness` and "
        "`safe_harmlessness`. Empty cells are not-yet-run.\n"
    )
    lines.append(
        "> Controlled comparison: every method shares the identical base model, "
        "data, rollout `n`, KL, optimizer and training-token budget; only the "
        "multi-objective credit rule differs. Scale-sensitive methods (LS, "
        "Tchebycheff, Lagrangian, Dynamic-HV) use the frozen calibration in "
        "`safe_calibration.json`.\n"
    )

    for tag, pretty in MODELS:
        lines.append(f"\n## {pretty}\n")
        lines.append("| Method | Helpfulness | Harmlessness | Hypervolume | #prompts |")
        lines.append("|---|---|---|---|---|")
        for method in METHODS:
            exp = f"{tag}_{method}"
            m = data.get(exp)
            label = METHOD_LABEL.get(method, method)
            if m is None:
                lines.append(f"| {label} | - | - | - | - |")
                continue
            help_ = m.get("safe_helpfulness")
            harm = m.get("safe_harmlessness")
            hv = m.get("hypervolume")
            n = m.get("num_prompts")
            lines.append(
                f"| {label} | {fmt(help_)} | {fmt(harm)} | {fmt(hv, 4)} | "
                f"{n if n is not None else '-'} |"
            )

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    n_done = sum(1 for tag, _ in MODELS for me in METHODS if f"{tag}_{me}" in data)
    print(f"[aggregate] {n_done} cells found; wrote {args.out}")


if __name__ == "__main__":
    main()
