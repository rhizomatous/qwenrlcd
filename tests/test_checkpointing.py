from __future__ import annotations

import json

import pytest

from qwenrlcd.checkpointing import (
    TRAINER_STATE_FILE,
    TrainerProgress,
    load_trainer_progress,
    resolve_resume_checkpoint,
)


def test_progress_round_trip(tmp_path) -> None:
    progress = TrainerProgress(
        epoch=3,
        next_batch=4,
        global_step=17,
        batches_per_epoch=9,
        gradient_accumulation_steps=2,
    )
    checkpoint = tmp_path / "checkpoint-17"
    progress.write(checkpoint)

    assert load_trainer_progress(checkpoint) == progress


def test_latest_uses_highest_complete_checkpoint(tmp_path) -> None:
    TrainerProgress(1, 0, 2, 2, 1).write(tmp_path / "checkpoint-2")
    TrainerProgress(5, 0, 10, 2, 1).write(tmp_path / "checkpoint-10")
    (tmp_path / "checkpoint-99").mkdir()
    (tmp_path / "checkpoint-nope").mkdir()

    assert resolve_resume_checkpoint("latest", tmp_path).name == "checkpoint-10"


def test_explicit_checkpoint_requires_trainer_state(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-5"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match=TRAINER_STATE_FILE):
        resolve_resume_checkpoint(checkpoint, tmp_path)


def test_invalid_progress_is_rejected(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / TRAINER_STATE_FILE).write_text(
        json.dumps(
            {
                "epoch": 0,
                "next_batch": 3,
                "global_step": 1,
                "batches_per_epoch": 3,
                "gradient_accumulation_steps": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="within the epoch"):
        load_trainer_progress(checkpoint)
