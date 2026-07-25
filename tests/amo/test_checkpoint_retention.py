from pathlib import Path

from verl.utils.checkpoint.checkpoint_manager import (
    GLOBAL_STEP_COMPLETED_MARKER,
    mark_global_step_checkpoint_complete,
    prune_global_step_checkpoints,
)


def _make_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"global_step_{step}"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "actor" / "model.pt").write_text("model")
    (checkpoint / "data.pt").write_text("dataloader")
    mark_global_step_checkpoint_complete(root, step)
    return checkpoint


def test_prune_global_step_checkpoints_removes_complete_old_directories(tmp_path: Path):
    step_10 = _make_checkpoint(tmp_path, 10)
    step_20 = _make_checkpoint(tmp_path, 20)
    step_30 = _make_checkpoint(tmp_path, 30)
    future_step = _make_checkpoint(tmp_path, 40)
    unrelated = tmp_path / "notes"
    unrelated.mkdir()

    removed = prune_global_step_checkpoints(tmp_path, max_ckpt_to_keep=2, current_global_step=30)

    assert removed == [str(step_10)]
    assert not step_10.exists()
    assert step_20.is_dir()
    assert step_30.is_dir()
    assert future_step.is_dir()
    assert unrelated.is_dir()


def test_prune_requires_a_completed_current_checkpoint(tmp_path: Path):
    step_10 = _make_checkpoint(tmp_path, 10)
    step_20 = _make_checkpoint(tmp_path, 20)

    removed = prune_global_step_checkpoints(tmp_path, max_ckpt_to_keep=1, current_global_step=30)

    assert removed == []
    assert step_10.is_dir()
    assert step_20.is_dir()


def test_partial_checkpoint_does_not_displace_an_older_usable_checkpoint(tmp_path: Path):
    step_10 = _make_checkpoint(tmp_path, 10)
    partial_step_20 = tmp_path / "global_step_20"
    (partial_step_20 / "actor").mkdir(parents=True)
    (partial_step_20 / "data.pt").write_bytes(b"truncated")
    step_30 = _make_checkpoint(tmp_path, 30)

    removed = prune_global_step_checkpoints(tmp_path, 2, current_global_step=30)

    assert removed == [str(partial_step_20)]
    assert step_10.is_dir()
    assert step_30.is_dir()


def test_completed_marker_is_written_atomically(tmp_path: Path):
    checkpoint = _make_checkpoint(tmp_path, 10)

    marker = mark_global_step_checkpoint_complete(tmp_path, 10)

    assert marker == str(checkpoint / GLOBAL_STEP_COMPLETED_MARKER)
    assert Path(marker).read_text() == "10\n"
    assert list(checkpoint.glob(f"{GLOBAL_STEP_COMPLETED_MARKER}.tmp.*")) == []


def test_prune_ignores_disabled_limits_and_checkpoint_symlinks(tmp_path: Path):
    step_10 = _make_checkpoint(tmp_path, 10)
    step_20 = _make_checkpoint(tmp_path, 20)
    symlink = tmp_path / "global_step_5"
    symlink.symlink_to(step_10, target_is_directory=True)

    assert prune_global_step_checkpoints(tmp_path, 0, current_global_step=20) == []
    removed = prune_global_step_checkpoints(tmp_path, 1, current_global_step=20)

    assert removed == [str(step_10)]
    assert symlink.is_symlink()
    assert step_20.is_dir()


def test_prune_ignores_noncanonical_step_aliases(tmp_path: Path):
    step_10 = _make_checkpoint(tmp_path, 10)
    step_20 = _make_checkpoint(tmp_path, 20)
    alias = tmp_path / "global_step_020"
    alias.mkdir()

    removed = prune_global_step_checkpoints(tmp_path, 1, current_global_step=20)

    assert removed == [str(step_10)]
    assert step_20.is_dir()
    assert alias.is_dir()
