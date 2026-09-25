from __future__ import annotations

from pathlib import Path
from typing import Any

COMPACT_MODEL_FILE = "trainable_model.safetensors"


def register_compact_model_state_hooks(accelerator: Any) -> None:
    """Keep Accelerate's optimizer/RNG state, but save only trainable model weights.

    A checkpoint without our compact file is an older full-model checkpoint;
    Accelerate loads that normally. This lets interrupted older runs resume.
    """
    if accelerator.num_processes != 1:
        raise ValueError("compact checkpoints currently support one process only")

    from safetensors.torch import load_file, save_file

    def trainable_parameters(model: Any) -> dict[str, Any]:
        unwrapped = accelerator.unwrap_model(model)
        parameters = {
            name: parameter
            for name, parameter in unwrapped.named_parameters()
            if parameter.requires_grad
        }
        if not parameters:
            raise ValueError("model has no trainable parameters to checkpoint")
        return parameters

    def save_hook(models: list[Any], weights: list[dict[str, Any]], output_dir: str) -> None:
        if len(models) != 1 or len(weights) != 1:
            raise ValueError("compact checkpoints require exactly one model")
        parameters = trainable_parameters(models[0])
        if accelerator.is_main_process:
            save_file(
                {
                    name: parameter.detach().cpu().contiguous()
                    for name, parameter in parameters.items()
                },
                str(Path(output_dir) / COMPACT_MODEL_FILE),
                metadata={"format": "qwenrlcd.trainable.v1"},
            )
        # Prevent Accelerate from serializing a second copy of model state. For
        # LoRA this is compact; for a full fine-tune it necessarily includes the
        # trainable backbone, but still avoids the redundant default model file.
        weights.clear()

    def load_hook(models: list[Any], input_dir: str) -> None:
        compact_file = Path(input_dir) / COMPACT_MODEL_FILE
        if not compact_file.is_file():
            return  # Backward compatibility with full-model Accelerate checkpoints.
        if len(models) != 1:
            raise ValueError("compact checkpoints require exactly one model")
        parameters = trainable_parameters(models[0])
        state = load_file(str(compact_file), device="cpu")
        if set(state) != set(parameters):
            raise ValueError(
                "compact checkpoint trainable keys do not match model: "
                f"missing={sorted(set(parameters) - set(state))} "
                f"unexpected={sorted(set(state) - set(parameters))}"
            )
        for name, parameter in parameters.items():
            if state[name].shape != parameter.shape:
                raise ValueError(f"compact checkpoint shape mismatch for {name}")
        result = accelerator.unwrap_model(models[0]).load_state_dict(state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) & set(parameters):
            raise ValueError("compact checkpoint failed to restore trainable parameters")
        # Optimizer, scheduler, dataloader, and RNG still load through Accelerate.
        models.clear()

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
