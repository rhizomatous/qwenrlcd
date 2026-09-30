from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

SEALED_SPLITS = ("test", "test_ood")


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"model artifact directory is empty: {root}")
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def initialize_or_resume_report(
    *,
    output: Path,
    run_dir: Path,
    config: dict[str, Any],
    development_metrics: dict[str, Any],
    attention_backend: str,
) -> dict[str, Any]:
    from .compare_capacity import metric_summary

    identity = {
        "model_id": config["model_id"],
        "run_dir": str(run_dir),
        "final_artifact_sha256": _sha256_tree(run_dir / "final"),
        "attention_backend": attention_backend,
    }
    if output.is_file():
        with output.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("selected_model") != identity:
            raise ValueError(
                "sealed report belongs to a different model or inference backend"
            )
        return report

    report = {
        "purpose": "one-shot evaluation of the frozen development-selected model",
        "selection_frozen_before_evaluation": True,
        "selected_model": identity,
        "development_validation": metric_summary(development_metrics),
        "splits": {},
        "complete": False,
    }
    # Commit the selected artifact identity before either sealed split is loaded.
    _atomic_write_json(output, report)
    return report


def _evaluate_split(
    *,
    split: str,
    config: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: Any,
    batch_size: int,
    progress_every: int,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from .data import DecisionCollator, DecisionDataset, decision_model_inputs
    from .hf_data import load_hf_bundles
    from .losses import decision_loss
    from .train import validation_metrics_with_reference

    bundles = load_hf_bundles(
        str(config["dataset_path"]),
        str(config.get("dataset_config", "core")),
        split,
    )
    dataset = DecisionDataset(
        bundles,
        tokenizer,
        max_length=int(config["max_length"]),
        max_choices=int(config["max_choices"]),
        max_questions=int(config["max_questions"]),
        shuffle=False,
        seed=int(config["seed"]),
        require_no_state_truncation=bool(
            config.get("require_no_state_truncation", False)
        ),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DecisionCollator(
            tokenizer,
            max_choices=int(config["max_choices"]),
            max_questions=int(config["max_questions"]),
            compact_attention_topology=(
                bool(config.get("compact_attention_topology", False))
                or config.get("attn_implementation", "eager") == "flex_attention"
            ),
        ),
        pin_memory=True,
    )
    rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    processed = 0
    with torch.inference_mode():
        for batch in dataloader:
            logits = model(**decision_model_inputs(batch, device))
            losses = decision_loss(
                logits,
                batch["targets"].to(device),
                batch["num_choices"].to(device),
                batch["question_mask"].to(device),
                batch["score_mask"].to(device),
                ce_weight=float(config["ce_weight"]),
                brier_weight=float(config["brier_weight"]),
                ordinal_rps_weight=float(config.get("ordinal_rps_weight", 0.0)),
            )
            probabilities = losses.probabilities.detach().float().cpu()
            targets = batch["targets"].float().cpu()
            question_mask = batch["question_mask"].cpu()
            num_choices = batch["num_choices"].cpu()
            bundle_losses = losses.per_bundle_total.detach().float().cpu()
            for row_index, bundle_id in enumerate(batch["ids"]):
                questions = []
                for question_index, valid in enumerate(
                    question_mask[row_index].tolist()
                ):
                    if not valid:
                        continue
                    choice_count = int(num_choices[row_index, question_index])
                    questions.append(
                        {
                            "type": batch["question_types"][row_index][question_index],
                            "probabilities": probabilities[
                                row_index, question_index, :choice_count
                            ].tolist(),
                            "target": targets[
                                row_index, question_index, :choice_count
                            ].tolist(),
                        }
                    )
                rows.append(
                    {
                        "id": bundle_id,
                        "source": batch["sources"][row_index],
                        "loss": float(bundle_losses[row_index]),
                        "questions": questions,
                    }
                )
            processed += len(batch["ids"])
            if processed % progress_every < len(batch["ids"]):
                elapsed = time.perf_counter() - started_at
                print(
                    f"[sealed] split={split} bundles={processed}/{len(dataset)} "
                    f"elapsed_minutes={elapsed / 60:.1f}",
                    flush=True,
                )
    metrics = validation_metrics_with_reference(rows)
    metrics["evaluation_seconds"] = time.perf_counter() - started_at
    return metrics


def _print_split(split: str, metrics: dict[str, Any]) -> None:
    print(
        f"[sealed] {split} bundles={metrics['bundles']} "
        f"questions={metrics['questions']} accuracy={metrics['accuracy']:.4f} "
        f"brier={metrics['brier']:.4f} kl={metrics['kl']:.4f} "
        f"ece={metrics['ece']:.4f}"
    )
    for question_type, values in metrics["slices"]["by_type"].items():
        print(
            f"[sealed] {split}/{question_type} n={values['count']} "
            f"brier={values['brier']:.4f} "
            f"kl={values['kl_divergence']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen model once on core test and test_ood"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--attention-backend", choices=("eager", "sdpa", "flex_attention"),
        default="sdpa",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.progress_every < 1:
        parser.error("batch-size and progress-every must be positive")

    from .train import load_config

    final_dir = args.run_dir / "final"
    metrics_path = args.run_dir / "validation_metrics.json"
    if not (final_dir / "decision_head.pt").is_file() or not metrics_path.is_file():
        parser.error("run directory needs a completed final model and validation metrics")
    config = load_config(args.run_dir / "training_config.json")
    with metrics_path.open(encoding="utf-8") as handle:
        development_history = json.load(handle)
    if not development_history:
        parser.error("development validation history is empty")
    output = args.output or args.run_dir / "sealed_evaluation.json"
    report = initialize_or_resume_report(
        output=output,
        run_dir=args.run_dir,
        config=config,
        development_metrics=development_history[-1],
        attention_backend=args.attention_backend,
    )
    if report.get("complete"):
        for split in SEALED_SPLITS:
            _print_split(split, report["splits"][split])
        print(f"[sealed] already complete: {output}")
        return

    import torch
    from transformers import AutoTokenizer

    from .model import DecisionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    tokenizer_dir = args.run_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir if tokenizer_dir.is_dir() else config["model_id"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    inference_config = {**config, "attn_implementation": args.attention_backend}
    model = DecisionModel.load_components(
        final_dir,
        model_id=config["model_id"],
        dtype=dtype,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        attn_implementation=args.attention_backend,
        flex_block_size=int(config.get("flex_block_size", 128)),
    ).to(device)
    model.eval()

    for split in SEALED_SPLITS:
        if split in report["splits"]:
            print(f"[sealed] split already complete, skipping: {split}")
            continue
        metrics = _evaluate_split(
            split=split,
            config=inference_config,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
        )
        report["splits"][split] = metrics
        _atomic_write_json(output, report)
        _print_split(split, metrics)

    report["complete"] = set(report["splits"]) == set(SEALED_SPLITS)
    _atomic_write_json(output, report)
    print(f"[sealed] report saved: {output}")


if __name__ == "__main__":
    main()
