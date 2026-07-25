from pathlib import Path
import os
from unittest.mock import patch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.checkpoint.checkpoint_manager import GLOBAL_STEP_COMPLETED_MARKER


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


class _ActorWorkerGroup:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.max_ckpt_to_keep = "not-called"

    def save_checkpoint(self, local_path, remote_path, global_step, max_ckpt_to_keep):
        self.max_ckpt_to_keep = max_ckpt_to_keep
        if self.fail:
            raise RuntimeError("injected checkpoint failure")
        actor = Path(local_path)
        actor.mkdir(parents=True)
        (actor / "model.pt").write_text(f"step={global_step}")


class _DataLoader:
    @staticmethod
    def state_dict():
        return {"next": 1}


def _make_existing_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"global_step_{step}"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "actor" / "model.pt").write_text("old")
    (checkpoint / "data.pt").write_text("old")
    (root / "latest_checkpointed_iteration.txt").write_text(str(step))
    return checkpoint


def _make_trainer(
    root: Path,
    actor_worker: _ActorWorkerGroup,
    *,
    global_step: int = 20,
    actor_keep: int = 1,
) -> RayPPOTrainer:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.global_steps = global_step
    trainer.use_critic = False
    trainer.actor_rollout_wg = actor_worker
    trainer.train_dataloader = _DataLoader()
    trainer.config = _AttrDict(
        trainer=_AttrDict(
            default_local_dir=str(root),
            default_hdfs_dir=None,
            remove_previous_ckpt_in_save=False,
            max_actor_ckpt_to_keep=actor_keep,
            max_critic_ckpt_to_keep=None,
        ),
        actor_rollout_ref=_AttrDict(
            actor=_AttrDict(checkpoint=_AttrDict(async_save=False)),
        ),
    )
    return trainer


def test_sync_save_rotates_complete_checkpoints_only_after_success(tmp_path: Path):
    old_checkpoint = _make_existing_checkpoint(tmp_path, 10)
    actor_worker = _ActorWorkerGroup()
    trainer = _make_trainer(tmp_path, actor_worker)

    trainer._save_checkpoint()

    assert actor_worker.max_ckpt_to_keep is None
    assert not old_checkpoint.exists()
    assert (tmp_path / "global_step_20" / "actor" / "model.pt").is_file()
    assert (tmp_path / "global_step_20" / "data.pt").is_file()
    assert (tmp_path / "global_step_20" / GLOBAL_STEP_COMPLETED_MARKER).is_file()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "20"


def test_failed_sync_save_preserves_last_resumable_checkpoint(tmp_path: Path):
    old_checkpoint = _make_existing_checkpoint(tmp_path, 10)
    actor_worker = _ActorWorkerGroup(fail=True)
    trainer = _make_trainer(tmp_path, actor_worker)

    try:
        trainer._save_checkpoint()
    except RuntimeError as error:
        assert str(error) == "injected checkpoint failure"
    else:
        raise AssertionError("checkpoint failure did not propagate")

    assert actor_worker.max_ckpt_to_keep is None
    assert (old_checkpoint / "actor" / "model.pt").is_file()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "10"
    assert not (tmp_path / "global_step_20").exists()


def test_sync_save_migrates_only_the_legacy_tracker_checkpoint(tmp_path: Path):
    old_checkpoint = _make_existing_checkpoint(tmp_path, 10)
    untracked_checkpoint = tmp_path / "global_step_15"
    (untracked_checkpoint / "actor").mkdir(parents=True)
    (untracked_checkpoint / "actor" / "model.pt").write_text("partial")
    (untracked_checkpoint / "data.pt").write_bytes(b"truncated")
    trainer = _make_trainer(
        tmp_path,
        _ActorWorkerGroup(),
        global_step=20,
        actor_keep=2,
    )

    trainer._save_checkpoint()

    assert old_checkpoint.is_dir()
    assert (old_checkpoint / GLOBAL_STEP_COMPLETED_MARKER).is_file()
    assert not untracked_checkpoint.exists()
    assert (tmp_path / "global_step_20" / GLOBAL_STEP_COMPLETED_MARKER).is_file()


def test_partial_dataloader_write_never_publishes_or_replaces_data_file(tmp_path: Path):
    old_checkpoint = _make_existing_checkpoint(tmp_path, 10)
    trainer = _make_trainer(tmp_path, _ActorWorkerGroup())

    def fail_after_partial_write(state_dict, file_object):
        file_object.write(b"truncated")
        raise OSError("injected dataloader write failure")

    with patch("verl.trainer.ppo.ray_trainer.torch.save", side_effect=fail_after_partial_write):
        try:
            trainer._save_checkpoint()
        except OSError as error:
            assert str(error) == "injected dataloader write failure"
        else:
            raise AssertionError("dataloader write failure did not propagate")

    assert (old_checkpoint / "actor" / "model.pt").is_file()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "10"
    assert not (tmp_path / "global_step_20" / "data.pt").exists()
    assert not (tmp_path / "global_step_20" / GLOBAL_STEP_COMPLETED_MARKER).exists()
    assert list((tmp_path / "global_step_20").glob("data.pt.tmp.*")) == []


def test_mixed_async_save_uses_legacy_limits_and_skips_driver_rotation(tmp_path: Path):
    old_checkpoint = _make_existing_checkpoint(tmp_path, 10)
    actor_worker = _ActorWorkerGroup()
    critic_worker = _ActorWorkerGroup()
    trainer = _make_trainer(tmp_path, actor_worker)
    trainer.use_critic = True
    trainer.critic_wg = critic_worker
    trainer.config["critic"] = _AttrDict(checkpoint=_AttrDict(async_save=True))
    trainer.config.trainer["max_critic_ckpt_to_keep"] = 2

    trainer._save_checkpoint()

    assert actor_worker.max_ckpt_to_keep == 1
    assert critic_worker.max_ckpt_to_keep == 2
    assert old_checkpoint.is_dir()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "10"
    assert (tmp_path / "global_step_20" / "actor" / "model.pt").is_file()
    assert (tmp_path / "global_step_20" / "critic" / "model.pt").is_file()


def test_tracker_replace_failure_preserves_previous_tracker_and_checkpoint(tmp_path: Path):
    old_checkpoint = _make_existing_checkpoint(tmp_path, 10)
    actor_worker = _ActorWorkerGroup()
    trainer = _make_trainer(tmp_path, actor_worker)

    real_replace = os.replace

    def fail_tracker_replace(source, destination):
        if str(destination).endswith("latest_checkpointed_iteration.txt"):
            raise OSError("injected tracker replace failure")
        return real_replace(source, destination)

    with patch("verl.trainer.ppo.ray_trainer.os.replace", side_effect=fail_tracker_replace):
        try:
            trainer._save_checkpoint()
        except OSError as error:
            assert str(error) == "injected tracker replace failure"
        else:
            raise AssertionError("tracker replace failure did not propagate")

    assert actor_worker.max_ckpt_to_keep is None
    assert (old_checkpoint / "actor" / "model.pt").is_file()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "10"
    assert (tmp_path / "global_step_20" / GLOBAL_STEP_COMPLETED_MARKER).is_file()
