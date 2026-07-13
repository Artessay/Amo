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
"""Append a finished training/eval run to the persistent experiment ledger.

Every experiment we run should be recorded here so results are never lost and
can be compared later for the paper. The ledger is an append-only JSONL file
(one record per run) plus an auto-regenerated human-readable Markdown table.

Two ways to use it:

1) Parse the final validation line straight out of a training log:

    python playground/log_experiment.py from-log \
        --log train_logs/hvpo_gpu1.log \
        --method hvpo --model qwen2.5-3b --dataset PKU-SafeRLHF \
        --notes "1 epoch, static ref, chebyshev"

2) Record arbitrary metrics directly (e.g. from the fast MOO benchmark):

    python playground/log_experiment.py add \
        --method hvpo --dataset moo_zdt1 \
        --metrics '{"HV": 0.6364, "IGD": 0.0258}' \
        --notes "20s CPU benchmark, 3 seeds"

The Markdown table at results/experiment_ledger/ledger.md is rebuilt on every
write so you always have an up-to-date at-a-glance comparison.
"""

import argparse
import json
import os
import re
from datetime import datetime

LEDGER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "experiment_ledger")
JSONL_PATH = os.path.join(LEDGER_DIR, "ledger.jsonl")
MD_PATH = os.path.join(LEDGER_DIR, "ledger.md")


def _parse_final_val_from_log(log_path: str) -> dict:
    """Extract the last validation metrics from a verl training log.

    Looks for the final ``training/global_step`` line and pulls every
    ``val-core/.../<name>:<value>`` and ``val-aux/.../<name>:<value>`` pair.
    """
    with open(log_path, "r", errors="ignore") as f:
        text = f.read()

    # Grab all val-core/val-aux key:value tokens from the whole file; keep the
    # LAST occurrence of each metric name (i.e. the final validation).
    metrics: dict[str, float] = {}
    step = None
    for m in re.finditer(r"training/global_step:(\d+)", text):
        step = int(m.group(1))
    for m in re.finditer(r"val-(?:core|aux)/[^\s:]*?/([A-Za-z_]+)/mean@\d+:([0-9.eE+-]+)", text):
        name, val = m.group(1), m.group(2)
        try:
            metrics[name] = float(val)
        except ValueError:
            continue
    return {"final_step": step, "metrics": metrics}


def _rebuild_markdown() -> None:
    """Regenerate ledger.md from ledger.jsonl (union of all metric columns)."""
    if not os.path.exists(JSONL_PATH):
        return
    records = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return

    # Collect the union of metric keys across all runs for stable columns.
    metric_keys: list[str] = []
    for r in records:
        for k in r.get("metrics", {}):
            if k not in metric_keys:
                metric_keys.append(k)

    fixed_cols = ["timestamp", "method", "model", "dataset", "final_step"]
    header = fixed_cols + metric_keys + ["notes"]

    lines = [
        "# Experiment Ledger",
        "",
        "Append-only record of every training/eval run. Generated from "
        "`ledger.jsonl` by `playground/log_experiment.py` — do not edit by hand.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in records:
        row = [
            str(r.get("timestamp", "")),
            str(r.get("method", "")),
            str(r.get("model", "")),
            str(r.get("dataset", "")),
            str(r.get("final_step", "")),
        ]
        for k in metric_keys:
            v = r.get("metrics", {}).get(k, "")
            row.append(f"{v:.4f}" if isinstance(v, (int, float)) else str(v))
        row.append(str(r.get("notes", "")).replace("|", "/"))
        lines.append("| " + " | ".join(row) + " |")

    with open(MD_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


def _append(record: dict) -> None:
    os.makedirs(LEDGER_DIR, exist_ok=True)
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    _rebuild_markdown()
    print(f"[ledger] appended run: {record.get('method')} / {record.get('dataset')} "
          f"-> {JSONL_PATH}")
    print(f"[ledger] table updated: {MD_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_log = sub.add_parser("from-log", help="parse final validation from a training log")
    p_log.add_argument("--log", required=True)
    p_log.add_argument("--method", required=True)
    p_log.add_argument("--model", default="")
    p_log.add_argument("--dataset", required=True)
    p_log.add_argument("--notes", default="")

    p_add = sub.add_parser("add", help="record arbitrary metrics directly")
    p_add.add_argument("--method", required=True)
    p_add.add_argument("--model", default="")
    p_add.add_argument("--dataset", required=True)
    p_add.add_argument("--metrics", required=True, help="JSON dict of metric->value")
    p_add.add_argument("--final-step", type=int, default=None)
    p_add.add_argument("--notes", default="")

    p_rebuild = sub.add_parser("rebuild", help="regenerate ledger.md from ledger.jsonl")

    args = ap.parse_args()

    # A deterministic-free timestamp is fine here: this is a CLI tool, not a
    # replayable workflow.
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    if args.cmd == "from-log":
        parsed = _parse_final_val_from_log(args.log)
        _append({
            "timestamp": ts,
            "method": args.method,
            "model": args.model,
            "dataset": args.dataset,
            "final_step": parsed["final_step"],
            "metrics": parsed["metrics"],
            "source_log": os.path.abspath(args.log),
            "notes": args.notes,
        })
    elif args.cmd == "add":
        _append({
            "timestamp": ts,
            "method": args.method,
            "model": args.model,
            "dataset": args.dataset,
            "final_step": args.final_step,
            "metrics": json.loads(args.metrics),
            "notes": args.notes,
        })
    elif args.cmd == "rebuild":
        _rebuild_markdown()
        print(f"[ledger] rebuilt {MD_PATH}")


if __name__ == "__main__":
    main()
