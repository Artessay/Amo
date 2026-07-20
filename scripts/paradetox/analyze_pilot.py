"""Compare held-out ParaDetox response sets produced by GRPO and HVPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from verl.trainer.amo_eval import compute_hypervolume


OBJECTIVES = ("detox_sta", "detox_sim", "detox_fluency")


def _step_files(directory: Path) -> dict[int, Path]:
    files = {int(p.stem): p for p in directory.glob("*.jsonl") if p.stem.isdigit()}
    if not files:
        raise FileNotFoundError(f"No numeric-step JSONL files in {directory}")
    return files


def _resolve_step(grpo_dir: Path, hvpo_dir: Path, step: str) -> int:
    grpo_steps = _step_files(grpo_dir)
    hvpo_steps = _step_files(hvpo_dir)
    if step == "final":
        grpo_final, hvpo_final = max(grpo_steps), max(hvpo_steps)
        if grpo_final != hvpo_final:
            raise ValueError(
                "Latest validation steps differ: "
                f"GRPO={grpo_final}, HVPO={hvpo_final}. Wait for both runs to finish or "
                "pass an explicit common --step."
            )
        return grpo_final
    selected = int(step)
    missing = [
        method
        for method, files in (("GRPO", grpo_steps), ("HVPO", hvpo_steps))
        if selected not in files
    ]
    if missing:
        raise FileNotFoundError(f"Step {selected} is missing for {', '.join(missing)}")
    return selected


def _load_step(directory: Path, selected: int) -> list[dict]:
    files = _step_files(directory)
    if selected not in files:
        raise FileNotFoundError(f"Step {selected} not found in {directory}; have {sorted(files)}")
    with files[selected].open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Step {selected} in {directory} is empty")
    wrong_steps = [row.get("step") for row in rows if "step" in row and int(row["step"]) != selected]
    if wrong_steps:
        raise ValueError(f"Rows in {files[selected]} contain a different step: {wrong_steps[0]}")
    return rows


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def _front_mask(points: np.ndarray) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        keep[i] = not any(_dominates(other, point) for j, other in enumerate(points) if i != j)
    return keep


def _group_rows(rows: list[dict], responses_per_prompt: int | None) -> list[list[dict]]:
    if responses_per_prompt is not None:
        if responses_per_prompt < 1:
            raise ValueError("responses_per_prompt must be positive")
        if len(rows) % responses_per_prompt:
            raise ValueError(
                f"{len(rows)} responses is not divisible by {responses_per_prompt} responses/prompt"
            )
        groups = [rows[i : i + responses_per_prompt] for i in range(0, len(rows), responses_per_prompt)]
        for index, group in enumerate(groups):
            prompts = {
                (str(row.get("input")), json.dumps(row.get("gts"), sort_keys=True)) for row in group
            }
            if len(prompts) != 1:
                raise ValueError(
                    f"Prompt group {index} contains multiple input/reference pairs; "
                    "validation ordering changed"
                )
        return groups

    groups: list[list[dict]] = []
    for row in rows:
        if not groups or row.get("input") != groups[-1][0].get("input"):
            groups.append([])
        groups[-1].append(row)
    return groups


def _analyze_rows(
    rows: list[dict], responses_per_prompt: int | None
) -> tuple[dict, dict[str, np.ndarray], list[tuple[str, str]]]:
    missing = [key for key in OBJECTIVES if any(key not in row for row in rows)]
    if missing:
        raise KeyError(f"Missing objective fields: {sorted(set(missing))}")

    groups = _group_rows(rows, responses_per_prompt)
    prompt_keys = [
        (str(group[0]["input"]), json.dumps(group[0].get("gts"), sort_keys=True)) for group in groups
    ]

    all_points = np.asarray([[float(row[key]) for key in OBJECTIVES] for row in rows], dtype=float)
    clipped = np.clip(all_points, 0.0, 1.0)
    prompt_hv = []
    prompt_joint = []
    prompt_linear = []
    prompt_objectives = []
    front_sizes = []
    conflict_rates = []
    duplicate_point_rates = []
    duplicate_output_rates = []
    centered_sta = []
    centered_sim = []
    for prompt_rows in groups:
        raw_points = np.asarray(
            [[float(row[key]) for key in OBJECTIVES] for row in prompt_rows], dtype=float
        )
        points = np.clip(raw_points, 0.0, 1.0)
        prompt_hv.append(compute_hypervolume(points, ref_point=[0.0, 0.0, 0.0]))
        prompt_joint.append(float(np.prod(points, axis=1).mean()))
        prompt_linear.append(float(raw_points.mean(axis=1).mean()))
        prompt_objectives.append(raw_points.mean(axis=0))
        centered_sta.extend(raw_points[:, 0] - raw_points[:, 0].mean())
        centered_sim.extend(raw_points[:, 1] - raw_points[:, 1].mean())

        unique_points = np.unique(points, axis=0)
        front_sizes.append(int(_front_mask(unique_points).sum()))
        incomparable = 0
        pair_count = 0
        for i, a in enumerate(unique_points):
            for b in unique_points[i + 1 :]:
                pair_count += 1
                incomparable += int(bool(np.any(a > b) and np.any(a < b)))
        conflict_rates.append(incomparable / pair_count if pair_count else 0.0)
        duplicate_point_rates.append(1.0 - len(unique_points) / len(points))
        duplicate_output_rates.append(
            1.0 - len({str(row.get("output", "")) for row in prompt_rows}) / len(prompt_rows)
        )

    corr = float(np.corrcoef(all_points[:, 0], all_points[:, 1])[0, 1]) if len(rows) > 1 else float("nan")
    within_prompt_corr = (
        float(np.corrcoef(centered_sta, centered_sim)[0, 1]) if len(rows) > 1 else float("nan")
    )
    metrics = {
        "num_prompts": len(groups),
        "num_responses": len(rows),
        "num_objective_values_clipped_for_hv": int(np.count_nonzero(all_points != clipped)),
        "mean_sta": float(all_points[:, 0].mean()),
        "mean_sim": float(all_points[:, 1].mean()),
        "mean_fluency": float(all_points[:, 2].mean()),
        "mean_linear_reward": float(all_points.mean(axis=1).mean()),
        "mean_joint_product": float(np.prod(clipped, axis=1).mean()),
        "mean_response_set_hv": float(np.mean(prompt_hv)),
        "mean_nondominated_per_prompt": float(np.mean(front_sizes)),
        "multi_point_front_rate": float(np.mean(np.asarray(front_sizes) >= 2)),
        "mean_incomparable_pair_rate": float(np.mean(conflict_rates)),
        "mean_duplicate_objective_point_rate": float(np.mean(duplicate_point_rates)),
        "mean_duplicate_output_rate": float(np.mean(duplicate_output_rates)),
        "sta_sim_pearson": corr,
        "within_prompt_sta_sim_pearson": within_prompt_corr,
        "mean_output_words": float(np.mean([len(row["output"].split()) for row in rows])),
    }
    objective_means = np.asarray(prompt_objectives)
    per_prompt = {
        "mean_response_set_hv": np.asarray(prompt_hv, dtype=float),
        "mean_joint_product": np.asarray(prompt_joint, dtype=float),
        "mean_linear_reward": np.asarray(prompt_linear, dtype=float),
        "mean_sta": objective_means[:, 0],
        "mean_sim": objective_means[:, 1],
        "mean_fluency": objective_means[:, 2],
    }
    return metrics, per_prompt, prompt_keys


def summarize(rows: list[dict], responses_per_prompt: int | None = None) -> dict:
    return _analyze_rows(rows, responses_per_prompt)[0]


def _validate_pairing(
    grpo_keys: list[tuple[str, str]], hvpo_keys: list[tuple[str, str]]
) -> None:
    if len(grpo_keys) != len(hvpo_keys):
        raise ValueError(f"Prompt counts differ: GRPO={len(grpo_keys)}, HVPO={len(hvpo_keys)}")
    mismatches = [i for i, (grpo, hvpo) in enumerate(zip(grpo_keys, hvpo_keys)) if grpo != hvpo]
    if mismatches:
        first = mismatches[0]
        raise ValueError(
            f"Prompt pairing differs at group {first}; use validation files from the same data/seed setup"
        )


def _paired_comparison(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    bootstrap_samples: int,
    bootstrap_seed: int,
    delta_label: str = "hvpo_minus_grpo",
) -> dict:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    num_prompts = len(next(iter(first.values())))
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, num_prompts, size=(bootstrap_samples, num_prompts))
    metrics = {}
    for key in first:
        delta = np.asarray(second[key]) - np.asarray(first[key])
        boot = delta[indices].mean(axis=1)
        metrics[key] = {
            delta_label: float(delta.mean()),
            "paired_bootstrap_95_ci": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
            "bootstrap_fraction_gt_zero": float(np.mean(boot > 0.0)),
            "positive_tied_negative_prompt_deltas": [
                int(np.count_nonzero(delta > 1e-12)),
                int(np.count_nonzero(np.abs(delta) <= 1e-12)),
                int(np.count_nonzero(delta < -1e-12)),
            ],
        }
    return {
        "num_paired_prompts": num_prompts,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "metrics": metrics,
    }


def _summarize_hvpo_rollouts(directory: Path, max_step: int, rollouts_per_prompt: int) -> dict:
    files = _step_files(directory)
    files = {step: path for step, path in files.items() if 0 < step <= max_step}
    if not files:
        raise FileNotFoundError(f"No HVPO rollout steps through {max_step} in {directory}")

    all_rows = []
    per_step = {}
    groups_with_positive = 0
    num_groups = 0
    for step, path in sorted(files.items()):
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        missing = [
            key
            for key in ("hv_contribution", "distance_penalty", "hybrid_rewards")
            if any(key not in row for row in rows)
        ]
        if missing:
            raise KeyError(f"Missing HVPO rollout fields in {path}: {sorted(set(missing))}")
        if any(int(row.get("step", step)) != step for row in rows):
            raise ValueError(f"Rows in {path} contain a different step")

        grouped = {}
        for row in rows:
            key = (str(row.get("input")), json.dumps(row.get("gts"), sort_keys=True))
            grouped.setdefault(key, []).append(row)
        groups = list(grouped.values())
        wrong_group_sizes = [len(group) for group in groups if len(group) != rollouts_per_prompt]
        if wrong_group_sizes:
            raise ValueError(
                f"Expected {rollouts_per_prompt} rollouts/prompt in {path}; "
                f"found group size {wrong_group_sizes[0]}"
            )
        contributions = np.asarray([float(row["hv_contribution"]) for row in rows])
        positive_per_group = [
            int(np.count_nonzero([float(row["hv_contribution"]) > 1e-12 for row in group]))
            for group in groups
        ]
        groups_with_positive += int(np.count_nonzero(np.asarray(positive_per_group) > 0))
        num_groups += len(groups)
        per_step[str(step)] = {
            "num_responses": len(rows),
            "hv_contribution_mean": float(contributions.mean()),
            "hv_contribution_std": float(contributions.std()),
            "hv_contribution_zero_fraction": float(np.mean(np.abs(contributions) <= 1e-12)),
            "mean_positive_contributions_per_prompt_group": float(np.mean(positive_per_group)),
        }
        all_rows.extend(rows)

    contributions = np.asarray([float(row["hv_contribution"]) for row in all_rows])
    distances = np.asarray([float(row["distance_penalty"]) for row in all_rows])
    hybrid = np.asarray([float(row["hybrid_rewards"]) for row in all_rows])
    return {
        "steps": sorted(files),
        "num_responses": len(all_rows),
        "num_prompt_groups": num_groups,
        "prompt_groups_with_positive_hv_contribution": groups_with_positive,
        "hv_contribution_mean": float(contributions.mean()),
        "hv_contribution_std": float(contributions.std()),
        "hv_contribution_min": float(contributions.min()),
        "hv_contribution_max": float(contributions.max()),
        "hv_contribution_zero_fraction": float(np.mean(np.abs(contributions) <= 1e-12)),
        "distance_penalty_mean": float(distances.mean()),
        "distance_penalty_nonzero_fraction": float(np.mean(np.abs(distances) > 1e-12)),
        "max_abs_hybrid_decomposition_error": float(np.max(np.abs(hybrid - contributions - distances))),
        "per_step": per_step,
    }


def _self_test() -> None:
    points = np.asarray([[0.9, 0.2, 1.0], [0.2, 0.9, 1.0], [0.1, 0.1, 0.5]])
    assert _front_mask(points).tolist() == [True, True, False]
    rows = []
    for point in points:
        rows.append(
            {
                "input": "same prompt",
                "output": "short output",
                **dict(zip(OBJECTIVES, point.tolist())),
            }
        )
    metrics = summarize(rows)
    assert metrics["num_prompts"] == 1
    assert metrics["mean_nondominated_per_prompt"] == 2.0
    assert metrics["mean_response_set_hv"] > 0
    duplicate_rows = rows + [dict(rows[0])]
    duplicate_metrics = summarize(duplicate_rows)
    assert duplicate_metrics["mean_nondominated_per_prompt"] == 2.0
    assert np.isclose(duplicate_metrics["mean_incomparable_pair_rate"], 1.0 / 3.0)

    grpo = {"metric": np.asarray([0.1, 0.2, 0.3])}
    hvpo = {"metric": np.asarray([0.2, 0.3, 0.4])}
    comparison = _paired_comparison(grpo, hvpo, bootstrap_samples=100, bootstrap_seed=0)
    assert np.isclose(comparison["metrics"]["metric"]["hvpo_minus_grpo"], 0.1)
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grpo-dir", type=Path)
    parser.add_argument("--hvpo-dir", type=Path)
    parser.add_argument("--step", default="final", help="Numeric validation step or 'final'.")
    parser.add_argument("--responses-per-prompt", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--baseline-step", type=int, default=0)
    parser.add_argument("--hvpo-rollout-dir", type=Path, default=None)
    parser.add_argument("--rollouts-per-prompt", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.grpo_dir is None or args.hvpo_dir is None:
        parser.error("--grpo-dir and --hvpo-dir are required unless --self-test is used")

    step = _resolve_step(args.grpo_dir, args.hvpo_dir, args.step)
    analyses = {}
    results = {}
    for method, directory in (("grpo", args.grpo_dir), ("hvpo", args.hvpo_dir)):
        rows = _load_step(directory, step)
        metrics, per_prompt, prompt_keys = _analyze_rows(rows, args.responses_per_prompt)
        analyses[method] = (per_prompt, prompt_keys)
        results[method] = {"step": step, **metrics}

    _validate_pairing(analyses["grpo"][1], analyses["hvpo"][1])
    comparison = _paired_comparison(
        analyses["grpo"][0],
        analyses["hvpo"][0],
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    results["paired_comparison"] = comparison

    if args.baseline_step != step:
        baseline_analyses = {}
        baseline_results = {}
        baseline_rows = {}
        for method, directory in (("grpo", args.grpo_dir), ("hvpo", args.hvpo_dir)):
            rows = _load_step(directory, args.baseline_step)
            metrics, per_prompt, prompt_keys = _analyze_rows(rows, args.responses_per_prompt)
            baseline_rows[method] = rows
            baseline_analyses[method] = (per_prompt, prompt_keys)
            baseline_results[method] = {"step": args.baseline_step, **metrics}
            _validate_pairing(prompt_keys, analyses[method][1])
        _validate_pairing(baseline_analyses["grpo"][1], baseline_analyses["hvpo"][1])

        control_fields = ("input", "output", "gts", *OBJECTIVES)
        baseline_exact_match = len(baseline_rows["grpo"]) == len(baseline_rows["hvpo"]) and all(
            all(grpo_row.get(key) == hvpo_row.get(key) for key in control_fields)
            for grpo_row, hvpo_row in zip(baseline_rows["grpo"], baseline_rows["hvpo"])
        )
        results["baseline_control"] = {
            "step": args.baseline_step,
            "methods_exactly_matched_on_inputs_outputs_and_objectives": baseline_exact_match,
            "grpo": baseline_results["grpo"],
            "hvpo": baseline_results["hvpo"],
        }

        changes = {}
        for method in ("grpo", "hvpo"):
            changes[method] = _paired_comparison(
                baseline_analyses[method][0],
                analyses[method][0],
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                delta_label=f"step_{step}_minus_step_{args.baseline_step}",
            )
        results["change_from_baseline"] = changes

        grpo_change = {
            key: analyses["grpo"][0][key] - baseline_analyses["grpo"][0][key]
            for key in analyses["grpo"][0]
        }
        hvpo_change = {
            key: analyses["hvpo"][0][key] - baseline_analyses["hvpo"][0][key]
            for key in analyses["hvpo"][0]
        }
        results["difference_in_differences"] = _paired_comparison(
            grpo_change,
            hvpo_change,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            delta_label="hvpo_change_minus_grpo_change",
        )

    hvpo_rollout_dir = args.hvpo_rollout_dir or args.hvpo_dir.parent / "rollouts"
    if step > 0 and hvpo_rollout_dir.is_dir():
        results["hvpo_training_signal"] = _summarize_hvpo_rollouts(
            hvpo_rollout_dir, step, args.rollouts_per_prompt
        )

    for method in ("grpo", "hvpo"):
        metrics = results[method]
        print(f"\n{method.upper()} @ step {metrics['step']}")
        for key, value in metrics.items():
            if key != "step":
                print(f"  {key}: {value}")

    print("\nHVPO - GRPO")
    for key, stats in comparison["metrics"].items():
        low, high = stats["paired_bootstrap_95_ci"]
        print(f"  {key}: {stats['hvpo_minus_grpo']:+.6f} (paired 95% CI [{low:+.6f}, {high:+.6f}])")

    if "change_from_baseline" in results:
        print(f"\nSTEP {step} - STEP {args.baseline_step}")
        delta_label = f"step_{step}_minus_step_{args.baseline_step}"
        for method in ("grpo", "hvpo"):
            stats = results["change_from_baseline"][method]["metrics"]["mean_response_set_hv"]
            low, high = stats["paired_bootstrap_95_ci"]
            print(f"  {method.upper()} HV: {stats[delta_label]:+.6f} (paired 95% CI [{low:+.6f}, {high:+.6f}])")
        did = results["difference_in_differences"]["metrics"]["mean_response_set_hv"]
        low, high = did["paired_bootstrap_95_ci"]
        print(
            "  HV difference-in-differences: "
            f"{did['hvpo_change_minus_grpo_change']:+.6f} "
            f"(paired 95% CI [{low:+.6f}, {high:+.6f}])"
        )

    if "hvpo_training_signal" in results:
        signal = results["hvpo_training_signal"]
        print("\nHVPO TRAINING SIGNAL")
        print(
            f"  hv_contribution: mean={signal['hv_contribution_mean']:.6f}, "
            f"std={signal['hv_contribution_std']:.6f}, "
            f"zero_fraction={signal['hv_contribution_zero_fraction']:.4f}"
        )
        print(
            "  prompt groups with a positive contribution: "
            f"{signal['prompt_groups_with_positive_hv_contribution']}/{signal['num_prompt_groups']}"
        )

    output = args.output or args.grpo_dir.parent.parent / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
