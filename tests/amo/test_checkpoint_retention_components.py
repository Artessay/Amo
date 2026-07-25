from pathlib import Path

from verl.utils.checkpoint.checkpoint_manager import (
    mark_global_step_checkpoint_complete,
    prune_global_step_component_checkpoints,
    prune_unusable_global_step_checkpoints,
)


def _make_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"global_step_{step}"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "actor" / "model.pt").write_text("model")
    (checkpoint / "data.pt").write_text("dataloader")
    mark_global_step_checkpoint_complete(root, step)
    return checkpoint


def test_component_prune_scans_checkpoints_created_before_resume(tmp_path: Path):
    step_10 = _make_checkpoint(tmp_path, 10)
    step_20 = _make_checkpoint(tmp_path, 20)
    step_30 = _make_checkpoint(tmp_path, 30)

    removed = prune_global_step_component_checkpoints(
        tmp_path,
        component="actor",
        max_ckpt_to_keep=1,
        current_global_step=30,
    )

    assert removed == [str(step_10 / "actor"), str(step_20 / "actor")]
    assert not (step_10 / "actor").exists()
    assert not (step_20 / "actor").exists()
    assert (step_30 / "actor").is_dir()
    assert (step_10 / "data.pt").is_file()
    assert (step_20 / "data.pt").is_file()


def test_component_prune_ignores_and_cleans_partial_checkpoints(tmp_path: Path):
    step_10 = _make_checkpoint(tmp_path, 10)
    partial_step_20 = tmp_path / "global_step_20"
    (partial_step_20 / "actor").mkdir(parents=True)
    step_30 = _make_checkpoint(tmp_path, 30)

    removed_components = prune_global_step_component_checkpoints(
        tmp_path, "actor", 2, current_global_step=30
    )
    removed_steps = prune_unusable_global_step_checkpoints(
        tmp_path, current_global_step=30, components=("actor",)
    )

    assert removed_components == [str(partial_step_20 / "actor")]
    assert removed_steps == [str(partial_step_20)]
    assert (step_10 / "actor").is_dir()
    assert (step_30 / "actor").is_dir()


def test_component_limits_preserve_the_larger_actor_or_critic_history(tmp_path: Path):
    checkpoints = []
    for step in (10, 20, 30):
        checkpoint = _make_checkpoint(tmp_path, step)
        (checkpoint / "critic").mkdir()
        (checkpoint / "critic" / "model.pt").write_text("critic")
        checkpoints.append(checkpoint)

    prune_global_step_component_checkpoints(tmp_path, "actor", 1, 30)
    prune_global_step_component_checkpoints(tmp_path, "critic", 2, 30)
    removed_steps = prune_unusable_global_step_checkpoints(
        tmp_path, current_global_step=30, components=("actor", "critic")
    )

    assert removed_steps == [str(checkpoints[0])]
    assert not checkpoints[0].exists()
    assert not (checkpoints[1] / "actor").exists()
    assert (checkpoints[1] / "critic").is_dir()
    assert (checkpoints[2] / "actor").is_dir()
    assert (checkpoints[2] / "critic").is_dir()


def test_component_prune_rejects_paths_outside_checkpoint_root(tmp_path: Path):
    _make_checkpoint(tmp_path, 10)
    for component in ("../actor", ".", "..", "actor/.."):
        try:
            prune_global_step_component_checkpoints(tmp_path, component, 1, 10)
        except ValueError:
            continue
        raise AssertionError(f"unsafe checkpoint component was accepted: {component!r}")


def test_component_prune_ignores_noncanonical_step_aliases(tmp_path: Path):
    step_10 = _make_checkpoint(tmp_path, 10)
    step_20 = _make_checkpoint(tmp_path, 20)
    alias = tmp_path / "global_step_020" / "actor"
    alias.mkdir(parents=True)

    removed = prune_global_step_component_checkpoints(tmp_path, "actor", 1, 20)

    assert removed == [str(step_10 / "actor")]
    assert (step_20 / "actor").is_dir()
    assert alias.is_dir()
