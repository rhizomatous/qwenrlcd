from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a parallel calibrated decision model")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, help="Override config output_dir")
    parser.add_argument(
        "--resume-from",
        help="Accelerate checkpoint directory, or 'latest' below output_dir",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="Save a checkpoint and exit after this optimizer step (for resume testing)",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    import torch
    from accelerate import Accelerator, DataLoaderConfiguration
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    from .checkpointing import (
        TrainerProgress,
        load_trainer_progress,
        prune_checkpoints,
        resolve_resume_checkpoint,
    )
    from .compact_checkpoint import register_compact_model_state_hooks
    from .data import DecisionCollator, DecisionDataset
    from .hf_data import load_configured_bundles
    from .losses import decision_loss
    from .metrics import summarize_validation
    from .model import DecisionModel

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            data_seed=seed,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], trust_remote_code=bool(config.get("trust_remote_code", True))
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    collator = DecisionCollator(
        tokenizer,
        max_choices=int(config["max_choices"]),
        max_questions=int(config["max_questions"]),
    )
    train_bundles = load_configured_bundles(config, "train")
    validation_bundles = load_configured_bundles(config, "validation")

    train_dataset = DecisionDataset(
        train_bundles,
        tokenizer,
        max_length=int(config["max_length"]),
        max_choices=int(config["max_choices"]),
        max_questions=int(config["max_questions"]),
        shuffle=bool(config.get("permute_training", True)),
        seed=seed,
        require_no_state_truncation=bool(config.get("require_no_state_truncation", False)),
    )
    validation_dataset = DecisionDataset(
        validation_bundles,
        tokenizer,
        max_length=int(config["max_length"]),
        max_choices=int(config["max_choices"]),
        max_questions=int(config["max_questions"]),
        shuffle=False,
        seed=seed,
        require_no_state_truncation=bool(config.get("require_no_state_truncation", False)),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(config["micro_batch_size"]),
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=int(config["micro_batch_size"]),
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
    )

    model = DecisionModel.from_pretrained(
        config["model_id"],
        max_choices=int(config["max_choices"]),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        attn_implementation=config.get("attn_implementation", "eager"),
    )
    if config.get("gradient_checkpointing", True):
        model.enable_gradient_checkpointing()

    lora = config.get("lora", {})
    if lora.get("enabled", False):
        model.attach_lora(
            rank=int(lora["rank"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora["dropout"]),
            target_modules=lora["target_modules"],
        )

    head_parameters = list(model.decision_head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    optimizer = AdamW(
        [
            {"params": backbone_parameters, "lr": float(config["learning_rate"])},
            {"params": head_parameters, "lr": float(config["head_learning_rate"])},
        ],
        weight_decay=float(config["weight_decay"]),
        fused=torch.cuda.is_available(),
    )

    update_steps_per_epoch = math.ceil(
        len(train_dataloader) / int(config["gradient_accumulation_steps"])
    )
    total_steps = update_steps_per_epoch * int(config["epochs"])
    if args.stop_after_step is not None and not 1 <= args.stop_after_step <= total_steps:
        raise ValueError(f"stop-after-step must be between 1 and {total_steps}")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * float(config["warmup_ratio"]))),
        num_training_steps=total_steps,
    )
    model, optimizer, train_dataloader, validation_dataloader, scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, validation_dataloader, scheduler
        )
    )
    register_compact_model_state_hooks(accelerator)

    output_dir = args.output_dir or Path(config["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir / "tokenizer")
        saved_config = {**config, "output_dir": str(output_dir)}
        (output_dir / "training_config.json").write_text(
            json.dumps(saved_config, indent=2) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()

    start_epoch = 0
    resume_batch = 0
    global_step = 0
    resume_checkpoint: Path | None = None
    if args.resume_from is not None:
        resume_checkpoint = resolve_resume_checkpoint(args.resume_from, output_dir)
        progress = load_trainer_progress(resume_checkpoint)
        if progress.batches_per_epoch != len(train_dataloader):
            raise ValueError(
                "checkpoint batches_per_epoch does not match the current dataloader: "
                f"{progress.batches_per_epoch} != {len(train_dataloader)}"
            )
        gradient_accumulation_steps = int(config["gradient_accumulation_steps"])
        if progress.gradient_accumulation_steps != gradient_accumulation_steps:
            raise ValueError(
                "checkpoint gradient accumulation does not match the config: "
                f"{progress.gradient_accumulation_steps} != "
                f"{gradient_accumulation_steps}"
            )
        if progress.global_step >= total_steps:
            raise ValueError(
                f"checkpoint step {progress.global_step} already reached total_steps={total_steps}"
            )
        accelerator.load_state(resume_checkpoint)
        start_epoch = progress.epoch
        resume_batch = progress.next_batch
        global_step = progress.global_step
        accelerator.print(
            f"resumed checkpoint={resume_checkpoint} epoch={start_epoch} "
            f"next_batch={resume_batch} step={global_step}/{total_steps}"
        )

    validation_history: list[dict[str, Any]] = []
    metrics_path = output_dir / "validation_metrics.json"
    if resume_checkpoint is not None and metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as handle:
            saved_history = json.load(handle)
        validation_history = [
            metrics for metrics in saved_history if int(metrics["step"]) <= global_step
        ]
    validation_every_epochs = int(config.get("validation_every_epochs", 1))
    if validation_every_epochs < 1:
        raise ValueError("validation_every_epochs must be at least one")
    save_every = int(config["save_every"])
    if save_every < 1:
        raise ValueError("save_every must be at least one")
    keep_last_checkpoints = int(config.get("keep_last_checkpoints", 2))
    if keep_last_checkpoints < 1:
        raise ValueError("keep_last_checkpoints must be at least one")

    def save_checkpoint(epoch: int, next_batch: int) -> None:
        next_epoch = epoch
        if next_batch >= len(train_dataloader):
            next_epoch += 1
            next_batch = 0
        checkpoint_dir = output_dir / f"checkpoint-{global_step}"
        incomplete_dir = output_dir / f".checkpoint-{global_step}.incomplete"
        if accelerator.is_main_process:
            if checkpoint_dir.exists() or incomplete_dir.exists():
                raise FileExistsError(
                    f"checkpoint target already exists: {checkpoint_dir} or {incomplete_dir}"
                )
            incomplete_dir.mkdir()
        accelerator.wait_for_everyone()
        accelerator.save_state(incomplete_dir)
        if accelerator.is_main_process:
            TrainerProgress(
                epoch=next_epoch,
                next_batch=next_batch,
                global_step=global_step,
                batches_per_epoch=len(train_dataloader),
                gradient_accumulation_steps=int(
                    config["gradient_accumulation_steps"]
                ),
            ).write(incomplete_dir)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            incomplete_dir.rename(checkpoint_dir)
            removed = prune_checkpoints(
                output_dir, keep_last=keep_last_checkpoints
            )
            if removed:
                accelerator.print(
                    "pruned old checkpoints: " + ", ".join(path.name for path in removed)
                )
        accelerator.wait_for_everyone()

    model.train()
    for epoch in range(start_epoch, int(config["epochs"])):
        train_dataset.set_epoch(epoch)
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)
        skipped_batches = resume_batch if epoch == start_epoch else 0
        active_dataloader = (
            accelerator.skip_first_batches(train_dataloader, skipped_batches)
            if skipped_batches
            else train_dataloader
        )
        for relative_batch_index, batch in enumerate(active_dataloader):
            batch_index = relative_batch_index + skipped_batches
            with accelerator.accumulate(model):
                logits = model(
                    input_ids=batch["input_ids"],
                    position_ids=batch["position_ids"],
                    tree_attention_mask=batch["tree_attention_mask"],
                    decision_indices=batch["decision_indices"],
                )
                losses = decision_loss(
                    logits,
                    batch["targets"],
                    batch["num_choices"],
                    batch["question_mask"],
                    ce_weight=float(config["ce_weight"]),
                    brier_weight=float(config["brier_weight"]),
                )
                accelerator.backward(losses.total)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), float(config["max_grad_norm"])
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % int(config["log_every"]) == 0:
                    accelerator.print(
                        f"epoch={epoch} step={global_step}/{total_steps} "
                        f"loss={losses.total.item():.4f} "
                        f"ce={losses.cross_entropy.item():.4f} "
                        f"brier={losses.brier.item():.4f}"
                    )
                should_stop = (
                    args.stop_after_step is not None
                    and global_step >= args.stop_after_step
                )
                if should_stop or (
                    global_step % save_every == 0 and global_step < total_steps
                ):
                    save_checkpoint(epoch, batch_index + 1)
                if should_stop:
                    accelerator.print(
                        f"stopped after step={global_step}; resume with "
                        f"--resume-from {output_dir / f'checkpoint-{global_step}'}"
                    )
                    return

        resume_batch = 0

        is_final_epoch = epoch + 1 == int(config["epochs"])
        if (epoch + 1) % validation_every_epochs != 0 and not is_final_epoch:
            continue

        model.eval()
        local_validation: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in validation_dataloader:
                logits = model(
                    input_ids=batch["input_ids"],
                    position_ids=batch["position_ids"],
                    tree_attention_mask=batch["tree_attention_mask"],
                    decision_indices=batch["decision_indices"],
                )
                losses = decision_loss(
                    logits,
                    batch["targets"],
                    batch["num_choices"],
                    batch["question_mask"],
                    ce_weight=float(config["ce_weight"]),
                    brier_weight=float(config["brier_weight"]),
                )
                probabilities = losses.probabilities.detach().float().cpu()
                targets = batch["targets"].float().cpu()
                question_mask = batch["question_mask"].cpu()
                num_choices = batch["num_choices"].cpu()
                bundle_losses = losses.per_bundle_total.detach().float().cpu()
                for row, bundle_id in enumerate(batch["ids"]):
                    questions = []
                    for index, valid in enumerate(question_mask[row].tolist()):
                        if not valid:
                            continue
                        choice_count = int(num_choices[row, index])
                        questions.append(
                            {
                                "type": batch["question_types"][row][index],
                                "probabilities": probabilities[
                                    row, index, :choice_count
                                ].tolist(),
                                "target": targets[row, index, :choice_count].tolist(),
                            }
                        )
                    local_validation.append(
                        {
                            "id": bundle_id,
                            "source": batch["sources"][row],
                            "loss": float(bundle_losses[row]),
                            "questions": questions,
                        }
                    )

        if accelerator.num_processes > 1:
            gathered: list[list[dict[str, Any]] | None] = [
                None for _ in range(accelerator.num_processes)
            ]
            torch.distributed.all_gather_object(gathered, local_validation)
            all_validation = [bundle for rank_bundles in gathered
                              for bundle in (rank_bundles or [])]
        else:
            all_validation = local_validation
        # Accelerate can pad the last distributed batch by repeating examples.
        unique_validation = {bundle["id"]: bundle for bundle in all_validation}
        if len(unique_validation) != len(validation_dataset):
            raise ValueError(
                "validation bundle IDs are missing or duplicated: "
                f"{len(unique_validation)} unique for {len(validation_dataset)} rows"
            )
        summary = summarize_validation(list(unique_validation.values()))
        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "step": global_step,
            "loss": summary["loss"],
            "accuracy": summary["accuracy"],
            "expected_accuracy": summary["expected_accuracy"],
            "brier": summary["brier"],
            "uniform_brier": summary["uniform_brier"],
            "nll": summary["negative_log_likelihood"],
            "ece": summary["expected_calibration_error"],
            "target_entropy": summary["target_entropy"],
            "kl": summary["kl_divergence"],
            "uniform_kl": summary["uniform_kl_divergence"],
            "js": summary["js_divergence"],
            "bundles": summary["bundles"],
            "questions": summary["count"],
            "slices": {
                key: value for key, value in summary.items()
                if key in {
                    "bundle_macro", "source_macro", "by_source", "by_type",
                    "by_choice_count", "by_question_count", "by_target_entropy",
                }
            },
        }
        validation_history.append(epoch_metrics)
        if accelerator.is_main_process:
            metrics_path.write_text(
                json.dumps(validation_history, indent=2) + "\n", encoding="utf-8"
            )
        accelerator.print(
            "validation "
            + " ".join(
                f"{key}={epoch_metrics[key]:.4f}"
                if isinstance(epoch_metrics[key], float)
                else f"{key}={epoch_metrics[key]}"
                for key in ("epoch", "step", "loss", "accuracy", "brier", "nll", "ece")
            )
        )
        model.train()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # The default keeps Accelerate's autocast/output-conversion forward wrapper.
        # Remove it so the original and freshly loaded models use identical precision.
        unwrapped = accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        unwrapped.eval()
        verification_batch = next(iter(validation_dataloader))
        with torch.inference_mode():
            reference_logits = unwrapped(
                input_ids=verification_batch["input_ids"],
                position_ids=verification_batch["position_ids"],
                tree_attention_mask=verification_batch["tree_attention_mask"],
                decision_indices=verification_batch["decision_indices"],
            ).float()

        unwrapped.save_components(output_dir / "final")
        metrics_path.write_text(
            json.dumps(validation_history, indent=2) + "\n", encoding="utf-8"
        )
        accelerator.print(f"saved final model components to {output_dir / 'final'}")

        reloaded = DecisionModel.load_components(
            output_dir / "final",
            model_id=config["model_id"],
            dtype=torch.bfloat16,
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            attn_implementation=config.get("attn_implementation", "eager"),
        ).to(accelerator.device)
        reloaded.eval()
        with torch.inference_mode():
            reloaded_logits = reloaded(
                input_ids=verification_batch["input_ids"],
                position_ids=verification_batch["position_ids"],
                tree_attention_mask=verification_batch["tree_attention_mask"],
                decision_indices=verification_batch["decision_indices"],
            ).float()
        torch.testing.assert_close(
            reloaded_logits, reference_logits, atol=0.001, rtol=0.001
        )
        reload_delta = float((reloaded_logits - reference_logits).abs().max())
        accelerator.print(f"reload verification passed max_logit_delta={reload_delta:.6f}")
        del reloaded


if __name__ == "__main__":
    main()
