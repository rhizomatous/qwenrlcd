from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

TRAINER_STATE_FILE = "trainer_state.json"


@dataclass(frozen=True, slots=True)
class TrainerProgress:
    epoch: int
    next_batch: int
    global_step: int
    batches_per_epoch: int
    gradient_accumulation_steps: int

    def __post_init__(self) -> None:
        if min(
            self.epoch,
            self.next_batch,
            self.global_step,
            self.batches_per_epoch,
            self.gradient_accumulation_steps,
        ) < 0:
            raise ValueError("trainer progress values cannot be negative")
        if self.batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.next_batch >= self.batches_per_epoch:
            raise ValueError("next_batch must be within the epoch")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> TrainerProgress:
        return cls(
            epoch=int(value["epoch"]),
            next_batch=int(value["next_batch"]),
            global_step=int(value["global_step"]),
            batches_per_epoch=int(value["batches_per_epoch"]),
            gradient_accumulation_steps=int(value["gradient_accumulation_steps"]),
        )

    def write(self, checkpoint_dir: str | Path) -> None:
        destination = Path(checkpoint_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / TRAINER_STATE_FILE).write_text(
            json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8"
        )


def load_trainer_progress(checkpoint_dir: str | Path) -> TrainerProgress:
    path = Path(checkpoint_dir) / TRAINER_STATE_FILE
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"invalid trainer state in {path}")
    return TrainerProgress.from_dict(value)


def _checkpoint_step(path: Path) -> int | None:
    prefix = "checkpoint-"
    if not path.is_dir() or not path.name.startswith(prefix):
        return None
    suffix = path.name[len(prefix) :]
    if not suffix.isdigit() or not (path / TRAINER_STATE_FILE).is_file():
        return None
    return int(suffix)


def resolve_resume_checkpoint(
    requested: str | Path, output_dir: str | Path
) -> Path:
    if str(requested) != "latest":
        checkpoint = Path(requested)
        if not checkpoint.is_dir():
            raise ValueError(f"resume checkpoint does not exist: {checkpoint}")
        if not (checkpoint / TRAINER_STATE_FILE).is_file():
            raise ValueError(f"resume checkpoint has no {TRAINER_STATE_FILE}: {checkpoint}")
        step = _checkpoint_step(checkpoint)
        progress = load_trainer_progress(checkpoint)
        if step is not None and step != progress.global_step:
            raise ValueError(
                f"checkpoint directory step {step} does not match trainer state "
                f"step {progress.global_step}"
            )
        return checkpoint

    candidates: list[tuple[int, Path]] = []
    for path in Path(output_dir).glob("checkpoint-*"):
        step = _checkpoint_step(path)
        if step is None:
            continue
        try:
            progress = load_trainer_progress(path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if progress.global_step == step:
            candidates.append((step, path))
    if not candidates:
        raise ValueError(f"no complete checkpoints found in {output_dir}")
    return max(candidates, key=lambda item: item[0])[1]
