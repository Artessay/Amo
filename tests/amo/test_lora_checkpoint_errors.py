from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from verl.utils.fsdp_utils import layered_summon_lora_params
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class _CheckpointManager:
    @staticmethod
    def save_checkpoint(**kwargs):
        return None


def test_rank_zero_lora_setup_failure_is_gathered_and_propagated(tmp_path):
    worker = SimpleNamespace(
        _is_actor=True,
        _is_offload_param=False,
        _is_lora=True,
        actor_module=SimpleNamespace(peft_config={"default": object()}),
        actor_module_fsdp=object(),
        checkpoint_manager=_CheckpointManager(),
        rank=0,
    )
    gathered_errors = []

    def gather_error(output, error):
        output[0] = error
        gathered_errors.append(error)

    save_checkpoint = ActorRolloutRefWorker.save_checkpoint.__wrapped__
    with (
        patch("verl.workers.fsdp_workers.dist.get_rank", return_value=0),
        patch("verl.workers.fsdp_workers.dist.get_world_size", return_value=1),
        patch("verl.workers.fsdp_workers.dist.all_gather_object", side_effect=gather_error),
        patch("verl.workers.fsdp_workers.dist.barrier"),
        patch(
            "verl.workers.fsdp_workers.os.makedirs",
            side_effect=OSError("injected mkdir failure"),
        ),
    ):
        try:
            save_checkpoint(
                worker,
                local_path=str(tmp_path / "global_step_20" / "actor"),
                global_step=20,
                max_ckpt_to_keep=None,
            )
        except RuntimeError as error:
            assert "Failed to prepare LoRA adapter checkpoint" in str(error)
            assert "injected mkdir failure" in str(error)
        else:
            raise AssertionError("rank-zero LoRA setup failure did not propagate")

    assert len(gathered_errors) == 1
    assert "injected mkdir failure" in gathered_errors[0]


def test_lora_preflight_failure_prevents_any_rank_from_entering_fsdp_collective(tmp_path):
    actor_module_fsdp = Mock()
    actor_module_fsdp.to.return_value = actor_module_fsdp
    worker = SimpleNamespace(
        _is_actor=True,
        _is_offload_param=False,
        _is_lora=True,
        actor_module=SimpleNamespace(peft_config={"default": object()}),
        actor_module_fsdp=actor_module_fsdp,
        checkpoint_manager=_CheckpointManager(),
        rank=1,
    )

    def gather_remote_setup_failure(output, error):
        output[0] = "rank 0: OSError: injected remote preflight failure"
        output[1] = error

    save_checkpoint = ActorRolloutRefWorker.save_checkpoint.__wrapped__
    with (
        patch("verl.workers.fsdp_workers.fsdp_version", return_value=1),
        patch("verl.workers.fsdp_workers.dist.get_rank", return_value=1),
        patch("verl.workers.fsdp_workers.dist.get_world_size", return_value=2),
        patch(
            "verl.workers.fsdp_workers.dist.all_gather_object",
            side_effect=gather_remote_setup_failure,
        ),
        patch("verl.workers.fsdp_workers.dist.barrier"),
        patch("verl.workers.fsdp_workers.layered_summon_lora_params") as summon_lora,
    ):
        try:
            save_checkpoint(
                worker,
                local_path=str(tmp_path / "global_step_20" / "actor"),
                global_step=20,
                max_ckpt_to_keep=None,
            )
        except RuntimeError as error:
            assert "Failed to prepare LoRA adapter checkpoint" in str(error)
            assert "injected remote preflight failure" in str(error)
        else:
            raise AssertionError("remote LoRA preflight failure did not propagate")

    actor_module_fsdp.to.assert_called_once()
    summon_lora.assert_not_called()


def test_layer_error_is_synchronized_before_the_next_fsdp_collective():
    layer_0 = Mock()
    layer_0.state_dict.return_value = {}
    layer_1 = Mock()
    layer_1.state_dict.return_value = {}
    fsdp_module = Mock()
    fsdp_module.named_modules.return_value = [
        ("_fsdp_wrapped_module.base_model.model.model.layers.0", layer_0),
        ("_fsdp_wrapped_module.base_model.model.model.layers.1", layer_1),
    ]
    gather_count = 0

    def gather_remote_layer_failure(output, value):
        nonlocal gather_count
        gather_count += 1
        output[0] = value
        if gather_count == 1:
            output[1] = {"error": None, "signature": value["signature"]}
        elif gather_count == 2:
            output[1] = "rank 1 module 'layers.0': RuntimeError: injected layer failure"
        else:
            raise AssertionError("a later collective was entered after a rank failed")

    device = SimpleNamespace(empty_cache=lambda: None)
    with (
        patch("verl.utils.fsdp_utils.fsdp_version", return_value=1),
        patch("verl.utils.fsdp_utils.dist.get_rank", return_value=0),
        patch("verl.utils.fsdp_utils.dist.get_world_size", return_value=2),
        patch(
            "verl.utils.fsdp_utils.dist.all_gather_object",
            side_effect=gather_remote_layer_failure,
        ),
        patch(
            "verl.utils.fsdp_utils.FSDP.summon_full_params",
            side_effect=lambda *args, **kwargs: nullcontext(),
        ) as summon_full_params,
        patch("verl.utils.fsdp_utils.get_torch_device", return_value=device),
        patch("peft.utils.save_and_load.get_peft_model_state_dict", return_value={}),
    ):
        try:
            layered_summon_lora_params(fsdp_module)
        except RuntimeError as error:
            assert "injected layer failure" in str(error)
        else:
            raise AssertionError("remote layer failure did not propagate")

    assert gather_count == 2
    summon_full_params.assert_called_once_with(layer_0, writeback=False)
