from __future__ import annotations

import os
import shlex
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
H2_WEIGHT_VARIANTS = {
    "math-lighteval": ("h2w200", "h2w020", "h2w002", "h2w110", "h2w101", "h2w011"),
    "news": (
        "h2w2000",
        "h2w0200",
        "h2w0020",
        "h2w0002",
        "h2w1100",
        "h2w1010",
        "h2w1001",
        "h2w0110",
        "h2w0101",
        "h2w0011",
    ),
    "rlla": ("h2w20", "h2w02"),
}


def run_entry(
    script: Path,
    *args: str,
    cwd: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["TRAINER_TRACE"] = "0"
    for key in ("TRAINER_VARIANT", "EXPERIMENT_NAME", "CHECKPOINT_DIR"):
        env.pop(key, None)
    if env_overrides:
        env.update(env_overrides)
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


def _checkpoint_from_dry_run(result: subprocess.CompletedProcess[str]) -> str:
    return next(
        line.removeprefix("Checkpoint: ")
        for line in result.stdout.splitlines()
        if line.startswith("Checkpoint: ")
    )


def test_ls_and_weighted_gdpo_h2_sweeps_dry_run_with_unique_checkpoints():
    method_weight_keys = {
        "ls": "amo_strategy.scalarize_config.weights",
        "gdpo_weighted": "algorithm.amo_objective_weights",
    }
    checkpoints: set[str] = set()

    for method, weight_key in method_weight_keys.items():
        for dataset, variants in H2_WEIGHT_VARIANTS.items():
            script = TRAINERS / method / f"run_{dataset}.sh"
            base = run_entry(script, "--dry-run")
            assert base.returncode == 0, base.stderr
            base_command = shlex.split(base.stdout.splitlines()[-1])
            assert not any(arg.startswith(f"{weight_key}=") for arg in base_command)
            assert _checkpoint_from_dry_run(base).endswith(f"qwen2.5-1.5b_{method}")
            checkpoints.add(_checkpoint_from_dry_run(base))

            for variant in variants:
                result = run_entry(
                    script,
                    "--dry-run",
                    env_overrides={"TRAINER_VARIANT": variant},
                )
                assert result.returncode == 0, f"{method}/{dataset}/{variant}: {result.stderr}"

                encoded = variant.removeprefix("h2w")
                weights = ",".join({"0": "0.0", "1": "0.5", "2": "1.0"}[digit] for digit in encoded)
                command = shlex.split(result.stdout.splitlines()[-1])
                assert f"{weight_key}=[{weights}]" in command
                assert f"qwen2.5-1.5b_{method}_{variant}" in result.stdout

                checkpoint = _checkpoint_from_dry_run(result)
                assert checkpoint not in checkpoints
                checkpoints.add(checkpoint)

    assert len(checkpoints) == 42


def test_h2_weight_variants_are_strictly_validated_in_dry_run():
    invalid_cases = (
        ("math-lighteval", "h2w20"),
        ("math-lighteval", "h2w300"),
        ("math-lighteval", "h2w100"),
        ("rlla", "h2w11"),
        ("rlla", "weights20"),
    )
    for method in ("ls", "gdpo_weighted"):
        for dataset, variant in invalid_cases:
            result = run_entry(
                TRAINERS / method / f"run_{dataset}.sh",
                "--dry-run",
                env_overrides={"TRAINER_VARIANT": variant},
            )
            assert result.returncode == 2, f"accepted invalid variant {method}/{dataset}/{variant}"
            assert "invalid" in result.stderr


def test_priority_queue_declares_order_and_isolates_variant_identity():
    queue = TRAINERS / "orchestration" / "run_priority_baselines.sh"
    syntax = subprocess.run(["bash", "-n", str(queue)], text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr

    source = queue.read_text()
    assert (
        'DEFAULT_METHODS="ls tchebycheff gdpo_weighted rvpo ctwa lagrangian '
        'fair_stable mgda gapo dynamic_hv nsga2 smsemoa"'
    ) in source
    assert 'MAX_ACTOR_CKPTS=${MAX_ACTOR_CKPTS:-3}' in source
    execution = source.index('log "=== priority baseline queue START')
    method_loop = source.index('for method in "${METHODS[@]}"', execution)
    dataset_loop = source.index('for dataset in "${DATASETS[@]}"', method_loop)
    variant_loop = source.index('for variant in "${CELL_VARIANTS[@]}"', dataset_loop)
    assert method_loop < dataset_loop < variant_loop
    assert 'TRAINER_VARIANT="$variant" EXPERIMENT_NAME="$experiment" CHECKPOINT_DIR="$checkpoint_dir"' in source
    assert 'method=%s dataset=%s experiment=%s variant=%s checkpoint=%s' in source
    assert "exec 9>&-" in source
