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
    args = parser.parse_args()
    config = load_config(args.config)

    import torch
    from accelerate import Accelerator
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    from .data import DecisionCollator, DecisionDataset, read_jsonl
    from .losses import decision_loss
    from .metrics import calibration_metrics
    from .model import DecisionModel

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        mixed_precision="bf16",
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
    train_dataset = DecisionDataset(
        read_jsonl(config["train_file"]),
        tokenizer,
        max_length=int(config["max_length"]),
        max_choices=int(config["max_choices"]),
        max_questions=int(config["max_questions"]),
        shuffle=True,
        seed=seed,
    )
    validation_dataset = DecisionDataset(
        read_jsonl(config["validation_file"]),
        tokenizer,
        max_length=int(config["max_length"]),
        max_choices=int(config["max_choices"]),
        max_questions=int(config["max_questions"]),
        shuffle=False,
        seed=seed,
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

    output_dir = Path(config["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir / "tokenizer")
        (output_dir / "training_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()

    global_step = 0
    validation_history: list[dict[str, float | int]] = []
    validation_every_epochs = int(config.get("validation_every_epochs", 1))
    if validation_every_epochs < 1:
        raise ValueError("validation_every_epochs must be at least one")
    model.train()
    for epoch in range(int(config["epochs"])):
        train_dataset.set_epoch(epoch)
        for batch in train_dataloader:
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
                if global_step % int(config["save_every"]) == 0:
                    accelerator.save_state(output_dir / f"checkpoint-{global_step}")

        is_final_epoch = epoch + 1 == int(config["epochs"])
        if (epoch + 1) % validation_every_epochs != 0 and not is_final_epoch:
            continue

        model.eval()
        validation_losses: list[float] = []
        validation_probabilities: list[list[float]] = []
        validation_targets: list[list[float]] = []
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
                gathered_loss, probabilities, targets, question_mask = (
                    accelerator.gather_for_metrics(
                        (
                            losses.total.detach().repeat(logits.shape[0]),
                            losses.probabilities,
                            batch["targets"],
                            batch["question_mask"],
                        )
                    )
                )
                validation_losses.extend(gathered_loss.float().cpu().tolist())
                probabilities = probabilities.float().cpu()
                targets = targets.float().cpu()
                question_mask = question_mask.cpu()
                validation_probabilities.extend(probabilities[question_mask].tolist())
                validation_targets.extend(targets[question_mask].tolist())

        calibration = calibration_metrics(validation_probabilities, validation_targets)
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "step": global_step,
            "loss": sum(validation_losses) / len(validation_losses),
            "accuracy": calibration.accuracy,
            "brier": calibration.brier,
            "nll": calibration.negative_log_likelihood,
            "ece": calibration.expected_calibration_error,
        }
        validation_history.append(epoch_metrics)
        accelerator.print(
            "validation "
            + " ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in epoch_metrics.items()
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
        (output_dir / "validation_metrics.json").write_text(
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
