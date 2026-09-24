from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .data import training_view_bundle
from .hf_data import select_question_type
from .metrics import calibration_metrics
from .schema import DecisionBundle, QuestionType


def choice_diagnostic_bundles(
    bundles: Sequence[DecisionBundle], *, training_epoch_view: int | None, seed: int
) -> list[DecisionBundle]:
    """Use short Choice-only inference, or the exact full packed training view."""
    if training_epoch_view is None:
        return select_question_type(bundles, QuestionType.CHOICE)
    selected = []
    for index in range(len(bundles)):
        permuted = training_view_bundle(
            bundles, index, epoch=training_epoch_view, seed=seed
        )
        if any(question.type is QuestionType.CHOICE for question in permuted.questions):
            selected.append(permuted)
    if not selected:
        raise ValueError("no Choice questions in the selected training view")
    return selected


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [row["target"] for row in rows]
    predictions = [row["probabilities"] for row in rows]
    uniform = [[1.0 / len(target)] * len(target) for target in targets]
    return {
        "count": len(rows),
        "uniform_random_accuracy": sum(1.0 / len(target) for target in targets) / len(rows),
        "model": asdict(calibration_metrics(predictions, targets)),
        "uniform": asdict(calibration_metrics(uniform, targets)),
    }


def _question_kl(row: dict[str, Any]) -> float:
    return sum(
        target * math.log(target / max(predicted, 1e-12))
        for target, predicted in zip(row["target"], row["probabilities"], strict=True)
        if target > 0
    )


def summarize_choice_predictions(
    rows: list[dict[str, Any]], *, examples_per_source: int = 2
) -> dict[str, Any]:
    if not rows:
        raise ValueError("choice diagnostic needs at least one prediction")
    if examples_per_source < 0:
        raise ValueError("examples_per_source must be non-negative")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cardinality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_cardinality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = row["source"]
        count = len(row["target"])
        if len(row["probabilities"]) != count or len(row["options"]) != count:
            raise ValueError("choice prediction, target, and options must align")
        by_source[source].append(row)
        by_cardinality[str(count)].append(row)
        by_source_cardinality[f"{source}/{count}"].append(row)

    examples = []
    for source, source_rows in sorted(by_source.items()):
        for row in sorted(source_rows, key=_question_kl, reverse=True)[:examples_per_source]:
            predicted_index = max(
                range(len(row["probabilities"])), key=row["probabilities"].__getitem__
            )
            target_index = max(range(len(row["target"])), key=row["target"].__getitem__)
            examples.append({
                "source": source,
                "bundle_id": row["bundle_id"],
                "question_id": row["question_id"],
                "state_excerpt": row["state_excerpt"],
                "instructions": row["instructions"],
                "kl": _question_kl(row),
                "predicted_key": row["options"][predicted_index]["key"],
                "target_key": row["options"][target_index]["key"],
                "options": [
                    {
                        "index": index,
                        **option,
                        "predicted": row["probabilities"][index],
                        "target": row["target"][index],
                    }
                    for index, option in enumerate(row["options"])
                ],
            })

    return {
        "overall": _metric_summary(rows),
        "by_source": {
            source: _metric_summary(group) for source, group in sorted(by_source.items())
        },
        "by_cardinality": {
            count: _metric_summary(group)
            for count, group in sorted(by_cardinality.items(), key=lambda item: int(item[0]))
        },
        "by_source_cardinality": {
            key: _metric_summary(group)
            for key, group in sorted(by_source_cardinality.items())
        },
        "worst_examples_by_source": examples,
    }


def compare_choice_rows(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    """Measure order sensitivity after aligning predictions by stable option keys."""
    def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        indexed = {(row["bundle_id"], row["question_id"]): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError("duplicate bundle/question identity in Choice predictions")
        return indexed

    original = index_rows(reference)
    reordered = index_rows(candidate)
    if not original or original.keys() != reordered.keys():
        raise ValueError("Choice views must contain the same nonempty question identities")

    comparisons: list[dict[str, Any]] = []
    for identity, first in original.items():
        second = reordered[identity]
        if first["source"] != second["source"]:
            raise ValueError(f"source differs for {identity}")

        def keyed(row: dict[str, Any], field: str) -> dict[str, float]:
            keys = [option["key"] for option in row["options"]]
            values = row[field]
            if len(keys) != len(values) or len(keys) != len(set(keys)):
                raise ValueError(f"invalid {field} option alignment for {identity}")
            return dict(zip(keys, values, strict=True))

        first_target = keyed(first, "target")
        second_target = keyed(second, "target")
        if first_target.keys() != second_target.keys() or any(
            not math.isclose(first_target[key], second_target[key], abs_tol=1e-6)
            for key in first_target
        ):
            raise ValueError(f"target differs after option-key alignment for {identity}")
        first_prediction = keyed(first, "probabilities")
        second_prediction = keyed(second, "probabilities")
        comparisons.append({
            "source": first["source"],
            "total_variation": 0.5 * sum(
                abs(first_prediction[key] - second_prediction[key])
                for key in first_prediction
            ),
            "argmax_agrees": max(
                first_prediction, key=lambda key: (first_prediction[key], key)
            ) == max(second_prediction, key=lambda key: (second_prediction[key], key)),
            "option_order_changed": [option["key"] for option in first["options"]]
            != [option["key"] for option in second["options"]],
        })

    def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "questions": len(items),
            "option_order_changed": sum(item["option_order_changed"] for item in items),
            "mean_total_variation": sum(item["total_variation"] for item in items)
            / len(items),
            "max_total_variation": max(item["total_variation"] for item in items),
            "argmax_agreement": sum(item["argmax_agrees"] for item in items)
            / len(items),
        }

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in comparisons:
        by_source[item["source"]].append(item)
    reference_order: dict[str, list[str]] = defaultdict(list)
    candidate_order: dict[str, list[str]] = defaultdict(list)
    for row in reference:
        reference_order[row["bundle_id"]].append(row["question_id"])
    for row in candidate:
        candidate_order[row["bundle_id"]].append(row["question_id"])
    multi_question = [
        bundle_id for bundle_id, questions in reference_order.items()
        if len(questions) > 1
    ]
    return {
        "overall": summary(comparisons),
        "by_source": {source: summary(items) for source, items in sorted(by_source.items())},
        "question_order": {
            "multi_question_bundles": len(multi_question),
            "changed_multi_question_bundles": sum(
                reference_order[bundle_id] != candidate_order[bundle_id]
                for bundle_id in multi_question
            ),
        },
    }


def _excerpt(value: Any, limit: int = 240) -> str:
    from .formatting import render_jsonlike

    rendered = render_jsonlike(value)
    return rendered if len(rendered) <= limit else rendered[:limit] + "…"


def predict_decision_rows(
    bundles: Sequence[DecisionBundle], *, model: Any, tokenizer: Any,
    config: dict[str, Any], batch_size: int, device: Any,
    question_type: QuestionType,
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    from .data import DecisionCollator, DecisionDataset, decision_model_inputs
    from .losses import mask_invalid_choices

    dataset = DecisionDataset(
        bundles,
        tokenizer,
        max_length=int(config["max_length"]),
        max_choices=int(config["max_choices"]),
        max_questions=int(config["max_questions"]),
        shuffle=False,
        seed=int(config["seed"]),
        require_no_state_truncation=bool(config.get("require_no_state_truncation", False)),
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
    )
    rows: list[dict[str, Any]] = []
    bundle_offset = 0
    with torch.inference_mode():
        for batch in dataloader:
            logits = model(**decision_model_inputs(batch, device))
            probabilities = torch.softmax(
                mask_invalid_choices(logits.float(), batch["num_choices"].to(device)),
                dim=-1,
            ).cpu()
            batch_bundles = bundles[bundle_offset : bundle_offset + len(batch["ids"])]
            for batch_index, bundle in enumerate(batch_bundles):
                for question_index, question in enumerate(bundle.questions):
                    if question.type is not question_type:
                        continue
                    rows.append({
                        "source": bundle.source or "unspecified",
                        "bundle_id": bundle.id,
                        "question_id": question.id,
                        "state_excerpt": _excerpt(bundle.state),
                        "instructions": _excerpt(question.instructions),
                        "options": [
                            {"key": option.key, "description": _excerpt(option.description, 120)}
                            for option in question.options
                        ],
                        "target": question.target_vector(),
                        "probabilities": probabilities[
                            batch_index, question_index, : len(question.options)
                        ].tolist(),
                    })
            bundle_offset += len(batch["ids"])
    return rows


def predict_choice_rows(
    bundles: Sequence[DecisionBundle], *, model: Any, tokenizer: Any,
    config: dict[str, Any], batch_size: int, device: Any,
) -> list[dict[str, Any]]:
    return predict_decision_rows(
        bundles, model=model, tokenizer=tokenizer, config=config,
        batch_size=batch_size, device=device, question_type=QuestionType.CHOICE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect saved Choice predictions and baselines")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--examples-per-source", type=int, default=2)
    parser.add_argument(
        "--training-epoch-view", type=int, metavar="EPOCH",
        help="Recreate an exact zero-based training epoch, including all packed branches",
    )
    parser.add_argument(
        "--permutation-check", action="store_true",
        help="Compare canonical, last training, and fresh seeded option/question orders",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.training_epoch_view is not None and args.split != "train":
        parser.error("training-epoch-view requires --split train")
    if args.permutation_check and (args.split != "train" or args.training_epoch_view is not None):
        parser.error("permutation-check requires --split train and no training-epoch-view")

    from .train import load_config

    config = load_config(args.run_dir / "training_config.json")
    if args.training_epoch_view is not None:
        if not bool(config.get("permute_training", True)):
            parser.error("this run did not permute its training questions")
        if not 0 <= args.training_epoch_view < int(config["epochs"]):
            parser.error(f"training epoch must be between 0 and {int(config['epochs']) - 1}")

    import torch
    from transformers import AutoTokenizer

    from .hf_data import load_configured_bundles
    from .model import DecisionModel
    source_bundles = load_configured_bundles(config, args.split)
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
    model = DecisionModel.load_components(
        args.run_dir / "final",
        model_id=config["model_id"],
        dtype=dtype,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        attn_implementation=config.get("attn_implementation", "eager"),
        flex_block_size=int(config.get("flex_block_size", 128)),
    ).to(device)
    model.eval()

    def rows_for_view(epoch: int | None) -> list[dict[str, Any]]:
        bundles = choice_diagnostic_bundles(
            source_bundles, training_epoch_view=epoch, seed=int(config["seed"])
        )
        return predict_choice_rows(
            bundles, model=model, tokenizer=tokenizer, config=config,
            batch_size=args.batch_size, device=device,
        )

    if args.permutation_check:
        last_epoch = int(config["epochs"]) - 1
        views = {
            "canonical": rows_for_view(None),
            "last_epoch_seeded": rows_for_view(last_epoch),
            "fresh_seeded_epoch": rows_for_view(last_epoch + 1),
        }
        report = {
            "view_epochs": {
                "canonical": None,
                "last_epoch_seeded": last_epoch,
                "fresh_seeded_epoch": last_epoch + 1,
            },
            "last_epoch_seeded_was_trained": bool(config.get("permute_training", True)),
            "views": {
                name: summarize_choice_predictions(rows, examples_per_source=0)
                for name, rows in views.items()
            },
            "canonical_vs_last_epoch_seeded": compare_choice_rows(
                views["canonical"], views["last_epoch_seeded"]
            ),
            "canonical_vs_fresh": compare_choice_rows(
                views["canonical"], views["fresh_seeded_epoch"]
            ),
        }
        output = args.output or args.run_dir / "choice_permutation_diagnostic.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for name, summary in report["views"].items():
            overall = summary["overall"]
            model_metrics = overall["model"]
            uniform = overall["uniform"]
            print(
                f"{name} n={overall['count']} accuracy={model_metrics['accuracy']:.3f} "
                f"brier={model_metrics['brier']:.3f} uniform_brier={uniform['brier']:.3f} "
                f"kl={model_metrics['kl_divergence']:.3f} "
                f"uniform_kl={uniform['kl_divergence']:.3f}"
            )
        for name in ("canonical_vs_last_epoch_seeded", "canonical_vs_fresh"):
            overall = report[name]["overall"]
            question_order = report[name]["question_order"]
            print(
                f"{name} mean_tv={overall['mean_total_variation']:.4f} "
                f"max_tv={overall['max_total_variation']:.4f} "
                f"argmax_agreement={overall['argmax_agreement']:.3f} "
                f"reordered_options={overall['option_order_changed']}/{overall['questions']} "
                "reordered_multi_question_bundles="
                f"{question_order['changed_multi_question_bundles']}/"
                f"{question_order['multi_question_bundles']}"
            )
        print(f"choice permutation diagnostic saved: {output}")
        return

    rows = rows_for_view(args.training_epoch_view)

    report = summarize_choice_predictions(
        rows, examples_per_source=args.examples_per_source
    )
    report["view"] = {
        "split": args.split,
        "training_epoch_view": args.training_epoch_view,
        "contains_all_training_branches": args.training_epoch_view is not None,
    }
    metrics_path = args.run_dir / "validation_metrics.json"
    if args.split == "validation" and metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as handle:
            logged_choice = json.load(handle)[-1]["slices"]["by_type"]["choice"]
        report["saved_validation_comparison"] = {
            name: report["overall"]["model"][name] - logged_choice[name]
            for name in ("accuracy", "brier", "kl_divergence")
        }
    view_name = (
        f"train_epoch{args.training_epoch_view}"
        if args.training_epoch_view is not None else args.split
    )
    output = args.output or args.run_dir / f"choice_{view_name}_diagnostic.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("choice diagnostic view:", report["view"])
    for name, summary in (("overall", report["overall"]), *report["by_source"].items()):
        model_metrics = summary["model"]
        uniform = summary["uniform"]
        print(
            f"{name} n={summary['count']} accuracy={model_metrics['accuracy']:.3f} "
            f"random_accuracy={summary['uniform_random_accuracy']:.3f} "
            f"brier={model_metrics['brier']:.3f} uniform_brier={uniform['brier']:.3f} "
            f"kl={model_metrics['kl_divergence']:.3f} "
            f"uniform_kl={uniform['kl_divergence']:.3f}"
        )
    if "saved_validation_comparison" in report:
        print("difference from saved validation:", report["saved_validation_comparison"])
    for example in report["worst_examples_by_source"]:
        predicted = next(
            option for option in example["options"]
            if option["key"] == example["predicted_key"]
        )
        print(
            f"worst source={example['source']} bundle={example['bundle_id']} "
            f"question={example['question_id']} choices={len(example['options'])} "
            f"predicted={example['predicted_key']}({predicted['predicted']:.3f}) "
            f"target={example['target_key']} kl={example['kl']:.3f}"
        )
    print(f"choice diagnostic saved: {output}")


if __name__ == "__main__":
    main()
