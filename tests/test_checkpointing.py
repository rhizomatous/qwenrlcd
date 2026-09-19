from __future__ import annotations

import json

import pytest

from qwenrlcd.checkpointing import (
    TRAINER_STATE_FILE,
    TrainerProgress,
    checkpoint_cleanup_candidates,
    load_trainer_progress,
    prune_checkpoints,
    require_free_space,
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


def test_pruning_preserves_latest_and_unrelated_run_files(tmp_path) -> None:
    (tmp_path / "training_config.json").write_text("{}", encoding="utf-8")
    for step in (5, 10, 15):
        checkpoint = tmp_path / f"checkpoint-{step}"
        TrainerProgress(step, 0, step, 1, 1).write(checkpoint)
        (checkpoint / "trainable_model.safetensors").write_bytes(b"weights")
    (tmp_path / "checkpoint-20").mkdir()  # Interrupted save, not resumable.
    (tmp_path / ".checkpoint-25.incomplete").mkdir()
    (tmp_path / "checkpoint-unrelated").mkdir()
    (tmp_path / "checkpoint-30").symlink_to(tmp_path / "checkpoint-15", target_is_directory=True)

    assert [path.name for path in checkpoint_cleanup_candidates(tmp_path, keep_last=2)] == [
        "checkpoint-5"
    ]
    assert [path.name for path in prune_checkpoints(tmp_path, keep_last=2)] == [
        "checkpoint-5"
    ]
    assert not (tmp_path / "checkpoint-5").exists()
    assert (tmp_path / "checkpoint-10").exists()
    assert (tmp_path / "checkpoint-15").exists()
    assert (tmp_path / "checkpoint-20").exists()
    assert (tmp_path / "checkpoint-30").is_symlink()
    assert (tmp_path / "checkpoint-unrelated").exists()
    assert {path.name for path in checkpoint_cleanup_candidates(
        tmp_path, keep_last=1, include_incomplete=True
    )} == {"checkpoint-10", "checkpoint-20", ".checkpoint-25.incomplete"}


def test_cannot_prune_all_without_final_artifacts(tmp_path) -> None:
    (tmp_path / "training_config.json").write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint-1"
    TrainerProgress(1, 0, 1, 1, 1).write(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match="cannot remove all checkpoints"):
        prune_checkpoints(tmp_path, keep_last=0)
    (tmp_path / "final").mkdir()
    (tmp_path / "final" / "decision_head.pt").write_bytes(b"head")
    (tmp_path / "final" / "decision_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "final" / "backbone").mkdir()
    (tmp_path / "final" / "backbone" / "adapter_config.json").write_text(
        "{}", encoding="utf-8"
    )
    (tmp_path / "final" / "backbone" / "adapter_model.safetensors").write_bytes(
        b"adapter"
    )
    (tmp_path / "validation_metrics.json").write_text("[]", encoding="utf-8")
    assert [path.name for path in prune_checkpoints(tmp_path, keep_last=0)] == [
        "checkpoint-1"
    ]
    assert (tmp_path / "final" / "decision_head.pt").exists()


def test_free_space_check_reports_shortage(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        "qwenrlcd.checkpointing.shutil.disk_usage",
        lambda _: SimpleNamespace(free=1 * 2**30),
    )
    with pytest.raises(RuntimeError, match="only 1.00 GiB free"):
        require_free_space(tmp_path, minimum_gib=2)
    require_free_space(tmp_path, minimum_gib=1)
