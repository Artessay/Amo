# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
import random
import re
import shutil
import stat
import uuid

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.trainer.config import CheckpointConfig
from verl.utils.device import get_device_name, get_torch_device


# Match the canonical name produced by the trainer. Reject aliases such as
# ``global_step_020`` so they can never displace ``global_step_20`` during
# retention while the tracker still points to the canonical path.
_GLOBAL_STEP_DIR_PATTERN = re.compile(r"^global_step_(0|[1-9]\d*)$")
GLOBAL_STEP_COMPLETED_MARKER = ".checkpoint_complete"


def fsync_directory(path: str) -> None:
    """Persist directory-entry updates made before this call."""
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def mark_global_step_checkpoint_complete(
    checkpoint_root: str,
    global_step: int,
    required_components: tuple[str, ...] = ("actor",),
) -> str:
    """Atomically mark a fully written global checkpoint as committed."""
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError(f"global_step must be a non-negative integer, got {global_step!r}")

    checkpoint_path = os.path.abspath(
        os.path.join(checkpoint_root, f"global_step_{global_step}")
    )
    if not os.path.isdir(checkpoint_path) or os.path.islink(checkpoint_path):
        raise FileNotFoundError(
            f"cannot commit missing or unsafe checkpoint directory: {checkpoint_path}"
        )
    if not required_components:
        raise ValueError("required_components must not be empty")
    for component in required_components:
        _validate_checkpoint_component(component)
    if not _is_nonempty_regular_file(os.path.join(checkpoint_path, "data.pt")):
        raise RuntimeError(f"cannot commit checkpoint without a complete data.pt: {checkpoint_path}")
    missing_components = [
        component
        for component in required_components
        if not _has_checkpoint_component(checkpoint_path, component)
    ]
    if missing_components:
        raise RuntimeError(
            f"cannot commit checkpoint with missing components {missing_components}: {checkpoint_path}"
        )

    marker_path = os.path.join(checkpoint_path, GLOBAL_STEP_COMPLETED_MARKER)
    temp_marker_path = f"{marker_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with open(temp_marker_path, "x", encoding="utf-8") as marker_file:
            marker_file.write(f"{global_step}\n")
            marker_file.flush()
            os.fsync(marker_file.fileno())
        os.replace(temp_marker_path, marker_path)
        fsync_directory(checkpoint_path)
    finally:
        if os.path.exists(temp_marker_path):
            os.remove(temp_marker_path)
    return marker_path


def _is_nonempty_regular_file(path: str) -> bool:
    try:
        file_stat = os.stat(path, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return stat.S_ISREG(file_stat.st_mode) and file_stat.st_size > 0


def _has_commit_evidence(checkpoint_path: str, global_step: int) -> bool:
    # A directory is committed only when both the atomic dataloader file and
    # the explicit marker are present. Legacy checkpoints are migrated from
    # the tracker by the trainer before rotation; arbitrary unmarked folders
    # are never guessed to be complete from their shape or file size.
    data_path = os.path.join(checkpoint_path, "data.pt")
    marker_path = os.path.join(checkpoint_path, GLOBAL_STEP_COMPLETED_MARKER)
    if not _is_nonempty_regular_file(data_path) or not _is_nonempty_regular_file(marker_path):
        return False
    try:
        with open(marker_path, encoding="utf-8") as marker_file:
            return int(marker_file.read().strip()) == global_step
    except (OSError, ValueError):
        return False


def _validate_checkpoint_component(component: str) -> None:
    if (
        component in {"", ".", ".."}
        or os.path.isabs(component)
        or os.path.basename(component) != component
        or os.path.normpath(component) != component
    ):
        raise ValueError(f"checkpoint component must be one directory name, got {component!r}")


def _has_checkpoint_component(checkpoint_path: str, component: str) -> bool:
    component_path = os.path.join(checkpoint_path, component)
    return os.path.isdir(component_path) and not os.path.islink(component_path)


def _scan_global_step_directories(
    checkpoint_root: str, current_global_step: int
) -> list[tuple[int, str]]:
    checkpoint_root = os.path.abspath(checkpoint_root)
    try:
        entries = list(os.scandir(checkpoint_root))
    except FileNotFoundError:
        return []

    checkpoints = []
    for entry in entries:
        match = _GLOBAL_STEP_DIR_PATTERN.fullmatch(entry.name)
        if match is None or not entry.is_dir(follow_symlinks=False):
            continue
        step = int(match.group(1))
        if step <= current_global_step:
            checkpoints.append((step, os.path.abspath(entry.path)))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def _remove_checkpoint_directories(checkpoints: list[tuple[int, str]]) -> list[str]:
    removed = []
    for _, path in sorted(checkpoints, key=lambda item: item[0]):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            continue
        removed.append(os.path.abspath(path))
    return removed


def prune_global_step_checkpoints(
    checkpoint_root: str,
    max_ckpt_to_keep: int,
    current_global_step: int,
    required_components: tuple[str, ...] = ("actor",),
) -> list[str]:
    """Keep the newest K usable global checkpoints and remove stale partials.

    Only checkpoints up to ``current_global_step`` participate in rotation. A
    committed, structurally complete current checkpoint must exist before
    anything is removed. Future-step directories, aliases, and symlinks are
    left untouched.

    Returns the absolute paths that were removed, oldest first.
    """
    if (
        isinstance(max_ckpt_to_keep, bool)
        or not isinstance(max_ckpt_to_keep, int)
        or max_ckpt_to_keep <= 0
    ):
        return []
    if not required_components:
        raise ValueError("required_components must not be empty")
    for component in required_components:
        _validate_checkpoint_component(component)

    checkpoints = _scan_global_step_directories(checkpoint_root, current_global_step)
    usable = []
    stale = []
    for step, path in checkpoints:
        is_usable = _has_commit_evidence(path, step) and all(
            _has_checkpoint_component(path, component) for component in required_components
        )
        (usable if is_usable else stale).append((step, path))

    if not any(step == current_global_step for step, _ in usable):
        return []
    return _remove_checkpoint_directories(stale + usable[:-max_ckpt_to_keep])


def prune_global_step_component_checkpoints(
    checkpoint_root: str,
    component: str,
    max_ckpt_to_keep: int,
    current_global_step: int,
) -> list[str]:
    """Remove an old actor or critic subtree while preserving its step folder."""
    if (
        isinstance(max_ckpt_to_keep, bool)
        or not isinstance(max_ckpt_to_keep, int)
        or max_ckpt_to_keep <= 0
    ):
        return []
    _validate_checkpoint_component(component)

    checkpoints = _scan_global_step_directories(checkpoint_root, current_global_step)
    usable = []
    stale = []
    for step, checkpoint_path in checkpoints:
        if not _has_checkpoint_component(checkpoint_path, component):
            continue
        component_path = os.path.join(checkpoint_path, component)
        target = usable if _has_commit_evidence(checkpoint_path, step) else stale
        target.append((step, component_path))

    if not any(step == current_global_step for step, _ in usable):
        return []

    removed = []
    for _, path in sorted(stale + usable[:-max_ckpt_to_keep], key=lambda item: item[0]):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            continue
        removed.append(os.path.abspath(path))
    return removed


def prune_unusable_global_step_checkpoints(
    checkpoint_root: str,
    current_global_step: int,
    components: tuple[str, ...],
) -> list[str]:
    """Remove stale partials and step folders with no retained components."""
    if not components:
        raise ValueError("components must not be empty")
    for component in components:
        _validate_checkpoint_component(component)

    checkpoints = _scan_global_step_directories(checkpoint_root, current_global_step)
    current_is_complete = any(
        step == current_global_step
        and _has_commit_evidence(path, step)
        and all(_has_checkpoint_component(path, component) for component in components)
        for step, path in checkpoints
    )
    if not current_is_complete:
        return []

    unusable = []
    for step, path in checkpoints:
        if step == current_global_step:
            continue
        has_retained_component = any(
            _has_checkpoint_component(path, component) for component in components
        )
        if not _has_commit_evidence(path, step) or not has_retained_component:
            unusable.append((step, path))
    return _remove_checkpoint_directories(unusable)


class BaseCheckpointManager:
    """
    A checkpoint manager that saves and loads the following states in a SPMD way:
    - model
    - optimizer
    - lr_scheduler
    - extra_states

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig | CheckpointConfig = None,
    ):
        self.checkpoint_config = checkpoint_config
        checkpoint_load_contents = checkpoint_config.get("load_contents", None) if checkpoint_config else None
        checkpoint_save_contents = checkpoint_config.get("save_contents", None) if checkpoint_config else None
        if checkpoint_load_contents is None:
            checkpoint_load_contents = ["model", "optimizer", "extra"]
        if checkpoint_save_contents is None:
            checkpoint_save_contents = ["model", "optimizer", "extra"]
        self.previous_global_step = None
        self.previous_saved_paths = []

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class
        self.checkpoint_load_contents = checkpoint_load_contents
        self.checkpoint_save_contents = checkpoint_save_contents

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    @property
    def should_save_model(self) -> bool:
        """
        Returns True if 'model' is in checkpoint_save_contents, indicating the model state should be saved.
        """
        return "model" in self.checkpoint_save_contents

    @property
    def should_save_optimizer(self) -> bool:
        """
        Returns True if 'optimizer' is in checkpoint_save_contents, indicating the optimizer state should be saved.
        """
        return "optimizer" in self.checkpoint_save_contents

    @property
    def should_save_extra(self) -> bool:
        """
        Returns True if 'extra' is in checkpoint_save_contents, indicating the extra state should be saved.
        """
        return "extra" in self.checkpoint_save_contents

    @property
    def should_save_hf_model(self) -> bool:
        """
        Returns True if 'hf_model' is in checkpoint_save_contents, indicating the model should be converted to hf
        model and saved.
        """
        return "hf_model" in self.checkpoint_save_contents

    @property
    def should_load_model(self) -> bool:
        """
        Returns True if 'model' is in checkpoint_load_contents, indicating the model state should be loaded.
        """
        return "model" in self.checkpoint_load_contents

    @property
    def should_load_optimizer(self) -> bool:
        """
        Returns True if 'optimizer' is in checkpoint_load_contents, indicating the optimizer state should be loaded.
        """
        return "optimizer" in self.checkpoint_load_contents

    @property
    def should_load_extra(self) -> bool:
        """
        Returns True if 'extra' is in checkpoint_load_contents, indicating the extra state should be loaded.
        """
        return "extra" in self.checkpoint_load_contents

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        raise NotImplementedError

    def save_checkpoint(
        self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep: int = None
    ):
        raise NotImplementedError

    @staticmethod
    def checkpath(local_path: str, hdfs_path: str):
        assert local_path is not None or hdfs_path is not None, "local_path and hdfs_path cannot be both None"
        return local_path is not None, local_path if local_path is not None else hdfs_path

    def remove_previous_save_local_path(self, path):
        if isinstance(path, str):
            path = [path]
        for p in path:
            abs_path = os.path.abspath(p)
            print(f"Checkpoint manager remove previous save local path: {abs_path}")
            if not os.path.exists(abs_path):
                continue
            shutil.rmtree(abs_path, ignore_errors=True)

    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        if get_device_name() != "cpu":
            rng_state[get_device_name()] = get_torch_device().get_rng_state()

        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        if get_device_name() != "cpu":
            get_torch_device().set_rng_state(rng_state[get_device_name()])


def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    """
    Return the most recent checkpoint directory based on a tracker file.

    Args:
        path (str): Base directory containing the checkpoint tracker.
        directory_format (str): Template for checkpoint subfolders with one
            placeholder for the iteration number (default "global_step_{}").

    Returns:
        str or None: Full path to the latest checkpoint directory, or
        None if the tracker or checkpoint folder is missing.
    """
    if path is None:
        return None

    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Checkpoint tracker file does not exist: {tracker_file}")
        return None

    with open(tracker_file, "rb") as f:
        iteration = int(f.read().decode())
    ckpt_path = os.path.join(path, directory_format.format(iteration))
    if not os.path.exists(ckpt_path):
        print("Checkpoint does not exist: %s", ckpt_path)
        return None

    print("Found checkpoint: %s", ckpt_path)
    return ckpt_path


def get_checkpoint_tracker_filename(root_path: str):
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    return os.path.join(root_path, "latest_checkpointed_iteration.txt")


def should_save_ckpt_esi(max_steps_duration: float, save_ckpt_duration: float = 60, redundant_time: float = 0) -> bool:
    """
    Determine if checkpoint should be saved based on capacity esi expiration.

    Args:
        max_steps_duration: Max estimated time (seconds) required to complete one training step
        save_ckpt_duration: Estimated time (seconds) required to save checkpoint (default: 60)
        redundant_time: Additional buffer time (seconds) for unexpected delays (default: 0)
    """
    exp_ts_mlp = os.getenv("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # vemlp
    exp_ts_aws = os.getenv("SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # aws
    if exp_ts_mlp:
        try:
            import time

            remaining = float(exp_ts_mlp) - time.time()
        except ValueError:
            return False
        return (
            remaining > 0
            and max_steps_duration > 0
            and remaining <= save_ckpt_duration + max_steps_duration + redundant_time
        )
    elif exp_ts_aws:
        from datetime import datetime, timedelta

        expiration_time = datetime.fromtimestamp(int(exp_ts_aws))
        time_difference = expiration_time - datetime.now()
        threshold_minutes = (save_ckpt_duration + max_steps_duration + redundant_time) / 60
        return time_difference < timedelta(minutes=threshold_minutes)
    else:
        return False
