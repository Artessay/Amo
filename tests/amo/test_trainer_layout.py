from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINERS = ROOT / "scripts" / "trainers"
METHOD_CONFIGS = {
    "grpo": ("grpo", "amo_vanilla"),
    "gdpo": ("gdpo", "amo_vanilla"),
    "hvpo": ("hvpo", "amo_hvpo"),
    "ls": ("grpo", "amo_scalarize"),
    "tchebycheff": ("grpo", "amo_scalarize"),
    "gdpo_weighted": ("gdpo_weighted", "amo_vanilla"),
    "rvpo": ("rvpo", "amo_vanilla"),
    "mgda": ("mgda", "amo_vanilla"),
    "gapo": ("gapo", "amo_vanilla"),
    "lagrangian": ("grpo", "amo_adaptive"),
    "fair_stable": ("grpo", "amo_adaptive"),
    "ctwa": ("grpo", "amo_adaptive"),
    "dynamic_hv": ("grpo", "amo_adaptive"),
    "nsga2": ("grpo", "amo_pareto"),
    "smsemoa": ("grpo", "amo_pareto"),
}
METHODS = set(METHOD_CONFIGS)


def run_entry(script: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["TRAINER_TRACE"] = "0"
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=cwd or ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_unified_method_layout_has_no_legacy_trainer_trees():
    for method in METHODS:
        method_dir = TRAINERS / method
        assert (method_dir / "method.sh").is_file()
        assert any(method_dir.glob("run_*.sh"))

    for legacy in ("grpo_trainer", "gdpo_trainer", "hvpo_trainer", "baseline_trainer"):
        assert not (ROOT / "scripts" / legacy).exists()


def test_every_public_entry_dry_runs():
    entries = sorted(
        script
        for method in METHODS
        for script in (TRAINERS / method).glob("run_*.sh")
    )
    entries.extend(sorted((TRAINERS / "hvpo" / "ablations").glob("run_*.sh")))

    assert entries
    for script in entries:
        result = run_entry(script, "--dry-run")
        assert result.returncode == 0, f"{script}:\nstdout={result.stdout}\nstderr={result.stderr}"
        assert "Checkpoint: " in result.stdout
        assert "Results: " in result.stdout
        assert "verl.trainer.main_ppo" in result.stdout


def test_artifact_identity_is_independent_of_calling_directory(tmp_path: Path):
    script = TRAINERS / "hvpo" / "run_math-lighteval.sh"
    result = run_entry(script, "1.5b", "50", "--dry-run", cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert (
        f"Checkpoint: {ROOT}/checkpoints/amo_math-lighteval/qwen2.5-1.5b_hvpo"
        in result.stdout
    )
    assert (
        f"Results: {ROOT}/results/MATH-LightEval/"
        "qwen2.5-1.5b_hvpo.{parquet,json}"
        in result.stdout
    )


def test_every_method_resolves_its_estimator_and_reward_manager():
    for method, (estimator, reward_manager) in METHOD_CONFIGS.items():
        script = TRAINERS / method / "run_pku-saferlhf.sh"
        result = run_entry(script, "1.5b", "--dry-run")
        assert result.returncode == 0, f"{method}: {result.stderr}"
        assert f"algorithm.adv_estimator={estimator}" in result.stdout
        assert f"reward_model.reward_manager={reward_manager}" in result.stdout


def test_safe_calibration_and_hvpo_variant_are_resolved():
    safe = run_entry(TRAINERS / "ls" / "run_pku-saferlhf.sh", "3b", "--dry-run")
    assert safe.returncode == 0, safe.stderr
    assert "amo_strategy.scalarize_config.normalize=affine" in safe.stdout
    assert "trainer.default_local_dir=" in safe.stdout

    variant = run_entry(
        TRAINERS / "hvpo" / "ablations" / "run_math-lighteval_lag3.sh",
        "1.5b",
        "--dry-run",
    )
    assert variant.returncode == 0, variant.stderr
    assert "qwen2.5-1.5b_hvpo_lag3" in variant.stdout
    assert "trainer.test_freq=3" in variant.stdout


def test_hydra_cannot_override_artifact_identity():
    script = TRAINERS / "grpo" / "run_math-lighteval.sh"
    result = run_entry(script, "--dry-run", "trainer.project_name=wrong")

    assert result.returncode == 2
    assert "changes artifact identity" in result.stderr
