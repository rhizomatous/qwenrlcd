from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .benchmark import timing_summary
from .data import DECISION_MODEL_INPUT_KEYS

LOSS_INPUT_KEYS = (
    "targets",
    "num_choices",
    "question_mask",
    "score_mask",
)
CASE_BATCH_SIZES = {
    "median": 2,
    "p95": 2,
    "maximum": 1,
}


def quantile_rank(length: int, quantile: float) -> int:
    if length < 1:
        raise ValueError("length must be positive")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    return round((length - 1) * quantile)


def select_length_cases(lengths: Sequence[int]) -> list[dict[str, Any]]:
    """Select distinct real rows near p50, p95, and the observed maximum."""
    if len(lengths) < 5 or any(length < 1 for length in lengths):
        raise ValueError("at least five positive packed lengths are required")
    ordered = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    specifications = (
        ("median", 0.50, 2),
        ("p95", 0.95, 2),
        ("maximum", 1.00, 1),
    )
    selected_by_name: dict[str, dict[str, Any]] = {}
    used: set[int] = set()
    # Reserve the extreme rows first, so a small fixture cannot reuse its maximum
    # for both p95 and maximum. Results are returned in the human-facing order.
    for name, quantile, batch_size in reversed(specifications):
        target_index = ordered[quantile_rank(len(ordered), quantile)]
        target_length = lengths[target_index]
        candidates = sorted(
            (index for index in range(len(lengths)) if index not in used),
            key=lambda index: (abs(lengths[index] - target_length), index),
        )
        indices = candidates[:batch_size]
        if len(indices) != batch_size:
            raise ValueError("not enough distinct rows for benchmark cases")
        used.update(indices)
        selected_by_name[name] = {
            "name": name,
            "quantile": quantile,
            "target_length": target_length,
            "indices": indices,
        }
    return [selected_by_name[name] for name, _, _ in specifications]


def reused_length_cases(report: dict[str, Any], dataset_length: int) -> list[dict[str, Any]]:
    """Validate and recover benchmark row selections from an earlier report."""
    raw_cases = report.get("selection")
    if not isinstance(raw_cases, list):
        raise ValueError("selection report has no selection list")
    by_name = {
        case.get("name"): case
        for case in raw_cases
        if isinstance(case, dict) and isinstance(case.get("name"), str)
    }
    if set(by_name) != set(CASE_BATCH_SIZES):
        raise ValueError("selection report must contain median, p95, and maximum cases")
    selections = []
    used: set[int] = set()
    for name in CASE_BATCH_SIZES:
        case = by_name[name]
        indices = case.get("indices")
        if (
            not isinstance(indices, list)
            or len(indices) != CASE_BATCH_SIZES[name]
            or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        ):
            raise ValueError(f"selection report has invalid {name} indices")
        if any(index < 0 or index >= dataset_length for index in indices):
            raise ValueError(f"selection report has out-of-range {name} indices")
        if used.intersection(indices):
            raise ValueError("selection report reuses a dataset row")
        used.update(indices)
        selections.append(
            {
                "name": name,
                "quantile": float(case["quantile"]),
                "target_length": int(case["target_length"]),
                "indices": indices,
                "expected_ids": case.get("ids"),
                "expected_lengths": case.get("individual_lengths"),
            }
        )
    return selections


def evenly_spaced_names(names: Sequence[str], count: int) -> list[str]:
    if count < 1:
        raise ValueError("count must be positive")
    ordered = sorted(set(names))
    if len(ordered) <= count:
        return ordered
    positions = (
        {round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)}
        if count > 1
        else {0}
    )
    return [ordered[position] for position in sorted(positions)]


def _scan_training_lengths(dataset: Any) -> list[int]:
    lengths: list[int] = []
    started = time.perf_counter()
    for index in range(len(dataset)):
        lengths.append(len(dataset[index]["input_ids"]))
        if (index + 1) % 25_000 == 0 or index + 1 == len(dataset):
            elapsed = time.perf_counter() - started
            print(
                f"scanned training rows={index + 1}/{len(dataset)} elapsed={elapsed / 60:.1f}m",
                flush=True,
            )
    return lengths


def _build_cases(
    dataset: Any,
    collator: Any,
    selections: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases = []
    for selection in selections:
        examples = [dataset[index] for index in selection["indices"]]
        batch = collator(examples)
        ids = list(batch["ids"])
        individual_lengths = [len(example["input_ids"]) for example in examples]
        if selection.get("expected_ids") not in (None, ids):
            raise ValueError(f"reused {selection['name']} selection IDs changed")
        if selection.get("expected_lengths") not in (None, individual_lengths):
            raise ValueError(f"reused {selection['name']} packed lengths changed")
        cases.append(
            {
                **{
                    key: value
                    for key, value in selection.items()
                    if not key.startswith("expected_")
                },
                "ids": ids,
                "sources": list(batch["sources"]),
                "individual_lengths": individual_lengths,
                "padded_length": int(batch["input_ids"].shape[1]),
                "questions": int(batch["question_mask"].sum().item()),
                "batch": batch,
            }
        )
    return cases


def _gradient_names(model: Any, sample_count: int) -> list[str]:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    head = [name for name in trainable if "decision_head" in name]
    lora = [name for name in trainable if "lora_" in name]
    if not head or not lora:
        raise ValueError("expected trainable decision-head and LoRA parameters")
    return sorted(set(head + evenly_spaced_names(lora, sample_count)))


def _valid_values(values: Any, batch: dict[str, Any], torch: Any) -> Any:
    slots = torch.arange(values.shape[-1], device=values.device).view(1, 1, -1)
    valid = batch["question_mask"].unsqueeze(-1) & (slots < batch["num_choices"].unsqueeze(-1))
    return values.detach()[valid].float().cpu()


def _centered_valid_logits(logits: Any, batch: dict[str, Any], torch: Any) -> Any:
    centered = []
    for row in range(logits.shape[0]):
        for question in range(logits.shape[1]):
            if not bool(batch["question_mask"][row, question]):
                continue
            choice_count = int(batch["num_choices"][row, question])
            values = logits[row, question, :choice_count].detach().float()
            centered.append(values - values.mean())
    if not centered:
        raise ValueError("batch contains no valid question logits")
    return torch.cat(centered).cpu()


def _capture_gradients(model: Any, names: Sequence[str], torch: Any) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    gradients = {}
    for name in names:
        gradient = parameters[name].grad
        if gradient is None:
            raise ValueError(f"selected trainable parameter has no gradient: {name}")
        gradients[name] = gradient.detach().float().cpu().clone()
    if not gradients or not all(torch.isfinite(value).all() for value in gradients.values()):
        raise ValueError("captured gradients must be finite and nonempty")
    return gradients


def _run_case(
    model: Any,
    cpu_batch: dict[str, Any],
    *,
    config: dict[str, Any],
    gradient_names: Sequence[str],
    model_bytes: int,
    warmups: int,
    repeats: int,
    seed: int,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .losses import decision_loss

    batch = {
        key: cpu_batch[key].to("cuda", non_blocking=False)
        for key in (*DECISION_MODEL_INPUT_KEYS, *LOSS_INPUT_KEYS)
        if key in cpu_batch
    }

    def prepare_iteration() -> None:
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward_backward() -> tuple[Any, Any]:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                **{key: batch[key] for key in DECISION_MODEL_INPUT_KEYS if key in batch}
            )
            losses = decision_loss(
                logits,
                batch["targets"],
                batch["num_choices"],
                batch["question_mask"],
                batch["score_mask"],
                ce_weight=float(config["ce_weight"]),
                brier_weight=float(config["brier_weight"]),
                ordinal_rps_weight=float(config.get("ordinal_rps_weight", 0.0)),
            )
        losses.total.backward()
        return logits, losses

    for _ in range(warmups):
        prepare_iteration()
        logits, losses = forward_backward()
        del logits, losses
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    timings_ms: list[float] = []
    peak_bytes = model_bytes
    capture: dict[str, Any] | None = None
    component_losses: dict[str, float] | None = None
    for repeat in range(repeats):
        prepare_iteration()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits, losses = forward_backward()
        torch.cuda.synchronize()
        timings_ms.append(1000 * (time.perf_counter() - start))
        peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated())
        if repeat == 0:
            capture = {
                "logits": _valid_values(logits, batch, torch),
                "centered_logits": _centered_valid_logits(logits, batch, torch),
                "probabilities": _valid_values(losses.probabilities, batch, torch),
                "gradients": _capture_gradients(model, gradient_names, torch),
                "loss": float(losses.total.detach().cpu()),
            }
            component_losses = {
                "total": capture["loss"],
                "cross_entropy": float(losses.cross_entropy.detach().cpu()),
                "brier": float(losses.brier.detach().cpu()),
                "ordinal_rps": float(losses.ordinal_rps.detach().cpu()),
            }
        del logits, losses
    if capture is None or component_losses is None:
        raise AssertionError("benchmark produced no measured pass")
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return {
        **timing_summary(timings_ms),
        "samples_ms": timings_ms,
        "losses": component_losses,
        "peak_allocated_mib": peak_bytes / 2**20,
        "peak_above_loaded_model_mib": max(0, peak_bytes - model_bytes) / 2**20,
    }, capture


def _gradient_comparison(
    reference: dict[str, Any], candidate: dict[str, Any], torch: Any
) -> dict[str, Any]:
    if reference.keys() != candidate.keys():
        raise ValueError("gradient captures have different parameters")
    reference_vector = torch.cat([reference[name].reshape(-1) for name in sorted(reference)])
    candidate_vector = torch.cat([candidate[name].reshape(-1) for name in sorted(candidate)])
    difference = candidate_vector - reference_vector
    reference_norm = float(torch.linalg.vector_norm(reference_vector))
    candidate_norm = float(torch.linalg.vector_norm(candidate_vector))
    difference_norm = float(torch.linalg.vector_norm(difference))
    denominator = max(reference_norm, 1e-12)
    if reference_norm == 0 and candidate_norm == 0:
        cosine = 1.0
    elif reference_norm == 0 or candidate_norm == 0:
        cosine = 0.0
    else:
        cosine = float(
            torch.dot(reference_vector, candidate_vector) / (reference_norm * candidate_norm)
        )
    return {
        "elements": int(reference_vector.numel()),
        "parameter_names": sorted(reference),
        "reference_l2": reference_norm,
        "candidate_l2": candidate_norm,
        "difference_l2": difference_norm,
        "relative_l2": difference_norm / denominator,
        "cosine_similarity": cosine,
        "max_absolute_delta": float(difference.abs().max()),
    }


def _run_backend(
    backend: str,
    cases: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    run_dir: Path,
    warmups: int,
    repeats: int,
    gradient_sample_count: int,
    flex_block_size: int,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from .model import DecisionModel

    model = DecisionModel.load_components(
        run_dir / "final",
        model_id=config["model_id"],
        dtype=torch.bfloat16,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        attn_implementation=backend,
        flex_block_size=flex_block_size,
        is_trainable=True,
    )
    if config.get("gradient_checkpointing", True):
        model.enable_gradient_checkpointing()
    model.to("cuda")
    model.train()
    torch.cuda.synchronize()
    model_bytes = torch.cuda.memory_allocated()
    gradient_names = _gradient_names(model, gradient_sample_count)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"backend={backend} loaded_model={model_bytes / 2**20:.1f}MiB "
        f"trainable_parameters={trainable_parameters:,} "
        f"sampled_gradient_tensors={len(gradient_names)}",
        flush=True,
    )
    results: dict[str, Any] = {
        "loaded_model_mib": model_bytes / 2**20,
        "trainable_parameters": trainable_parameters,
        "gradient_parameter_names": gradient_names,
        "cases": {},
    }
    captures: dict[str, dict[str, Any]] = {}
    for case_index, case in enumerate(cases):
        result, capture = _run_case(
            model,
            case["batch"],
            config=config,
            gradient_names=gradient_names,
            model_bytes=model_bytes,
            warmups=warmups,
            repeats=repeats,
            seed=int(config["seed"]) + case_index,
            torch=torch,
        )
        results["cases"][case["name"]] = result
        captures[case["name"]] = capture
        print(
            f"  {case['name']} batch={len(case['ids'])} length={case['padded_length']} "
            f"forward_backward_p50={result['p50_ms']:.1f}ms "
            f"peak_extra={result['peak_above_loaded_model_mib']:.1f}MiB "
            f"loss={result['losses']['total']:.6f}",
            flush=True,
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results, captures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare attention backends on real BF16 training forward/backward passes"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference-backend",
        choices=("eager", "sdpa", "flex_attention"),
        default="eager",
    )
    parser.add_argument(
        "--candidate-backend",
        choices=("eager", "sdpa", "flex_attention"),
        default="sdpa",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gradient-sample-count", type=int, default=8)
    parser.add_argument("--flex-block-size", type=int, choices=(64, 128), default=128)
    parser.add_argument("--loss-atol", type=float, default=0.01)
    parser.add_argument("--loss-rtol", type=float, default=0.01)
    parser.add_argument("--probability-atol", type=float, default=0.005)
    parser.add_argument("--centered-logit-atol", type=float, default=0.10)
    parser.add_argument("--gradient-cosine-min", type=float, default=0.995)
    parser.add_argument("--gradient-relative-l2-max", type=float, default=0.10)
    parser.add_argument(
        "--selection-report",
        type=Path,
        help="Reuse and validate row indices from an earlier benchmark instead of rescanning",
    )
    parser.add_argument(
        "--allow-performance-failure",
        action="store_true",
        help="Write results and succeed when only the performance gate fails",
    )
    args = parser.parse_args()
    if args.warmups < 0 or min(args.repeats, args.gradient_sample_count) < 1:
        parser.error(
            "repeats and gradient sample count must be positive; warmups cannot be negative"
        )
    if args.reference_backend == args.candidate_backend:
        parser.error("reference and candidate backends must differ")
    if not (args.run_dir / "final" / "decision_head.pt").is_file():
        parser.error(f"no saved decision model in {args.run_dir / 'final'}")
    output_path = args.output or (
        args.run_dir / f"{args.candidate_backend}_training_benchmark.json"
    )
    if output_path.exists():
        parser.error(f"refusing to overwrite existing benchmark: {output_path}")

    import torch
    from transformers import AutoTokenizer

    from .data import DecisionCollator, DecisionDataset
    from .hf_data import load_configured_bundles
    from .train import load_config

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required for this training benchmark")
    if not torch.cuda.is_bf16_supported():
        parser.error("BF16-capable CUDA hardware is required")
    config = load_config(args.run_dir / "training_config.json")
    tokenizer_dir = args.run_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir if tokenizer_dir.is_dir() else config["model_id"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bundles = load_configured_bundles(config, "train")
    dataset = DecisionDataset(
        bundles,
        tokenizer,
        max_length=int(config["max_length"]),
        max_choices=int(config["max_choices"]),
        max_questions=int(config["max_questions"]),
        shuffle=bool(config.get("permute_training", True)),
        seed=int(config["seed"]),
        require_no_state_truncation=bool(config.get("require_no_state_truncation", False)),
    )
    dataset.set_epoch(0)
    if args.selection_report is None:
        print(
            f"scanning {len(dataset)} real epoch-0 training rows for length cases",
            flush=True,
        )
        lengths = _scan_training_lengths(dataset)
        selections = select_length_cases(lengths)
    else:
        with args.selection_report.open(encoding="utf-8") as handle:
            selection_report = json.load(handle)
        selections = reused_length_cases(selection_report, len(dataset))
        print(
            f"reusing validated row indices from {args.selection_report}; full length scan skipped",
            flush=True,
        )
    collator = DecisionCollator(
        tokenizer,
        max_choices=int(config["max_choices"]),
        max_questions=int(config["max_questions"]),
        compact_attention_topology=(
            args.reference_backend == "flex_attention"
            or args.candidate_backend == "flex_attention"
        ),
    )
    cases = _build_cases(dataset, collator, selections)
    for case in cases:
        print(
            f"selected {case['name']} ids={case['ids']} sources={case['sources']} "
            f"lengths={case['individual_lengths']} padded={case['padded_length']} "
            f"questions={case['questions']}",
            flush=True,
        )

    backend_results: dict[str, Any] = {}
    backend_captures: dict[str, dict[str, dict[str, Any]]] = {}
    for backend in (args.reference_backend, args.candidate_backend):
        backend_results[backend], backend_captures[backend] = _run_backend(
            backend,
            cases,
            config=config,
            run_dir=args.run_dir,
            warmups=args.warmups,
            repeats=args.repeats,
            gradient_sample_count=args.gradient_sample_count,
            flex_block_size=args.flex_block_size,
            torch=torch,
        )

    comparisons = []
    for case in cases:
        name = case["name"]
        reference = backend_results[args.reference_backend]["cases"][name]
        candidate = backend_results[args.candidate_backend]["cases"][name]
        reference_capture = backend_captures[args.reference_backend][name]
        candidate_capture = backend_captures[args.candidate_backend][name]
        logit_delta = float(
            (reference_capture["logits"] - candidate_capture["logits"]).abs().max()
        )
        centered_logit_delta = float(
            (
                reference_capture["centered_logits"]
                - candidate_capture["centered_logits"]
            ).abs().max()
        )
        probability_delta = float(
            (
                reference_capture["probabilities"]
                - candidate_capture["probabilities"]
            ).abs().max()
        )
        loss_delta = abs(reference_capture["loss"] - candidate_capture["loss"])
        loss_tolerance = args.loss_atol + args.loss_rtol * abs(reference_capture["loss"])
        gradient = _gradient_comparison(
            reference_capture["gradients"], candidate_capture["gradients"], torch
        )
        equivalence_passed = (
            loss_delta <= loss_tolerance
            and probability_delta <= args.probability_atol
            and centered_logit_delta <= args.centered_logit_atol
            and gradient["cosine_similarity"] >= args.gradient_cosine_min
            and gradient["relative_l2"] <= args.gradient_relative_l2_max
        )
        comparison = {
            "name": name,
            "reference_backend": args.reference_backend,
            "candidate_backend": args.candidate_backend,
            "reference_over_candidate_speedup": (
                reference["p50_ms"] / candidate["p50_ms"]
            ),
            "candidate_peak_extra_mib_delta": (
                candidate["peak_above_loaded_model_mib"]
                - reference["peak_above_loaded_model_mib"]
            ),
            "loss_absolute_delta": loss_delta,
            "loss_tolerance": loss_tolerance,
            "max_valid_logit_delta": logit_delta,
            "max_centered_logit_delta": centered_logit_delta,
            "max_probability_delta": probability_delta,
            "gradient": gradient,
            "equivalence_passed": equivalence_passed,
        }
        comparisons.append(comparison)
        print(
            f"compare {name} reference={args.reference_backend} "
            f"candidate={args.candidate_backend} "
            f"speedup={comparison['reference_over_candidate_speedup']:.2f}x "
            f"peak_delta={comparison['candidate_peak_extra_mib_delta']:+.1f}MiB "
            f"loss_delta={loss_delta:.6f} raw_logit_delta={logit_delta:.6f} "
            f"centered_logit_delta={centered_logit_delta:.6f} "
            f"probability_delta={probability_delta:.6f} "
            f"grad_cos={gradient['cosine_similarity']:.6f} "
            f"grad_rel_l2={gradient['relative_l2']:.6f} "
            f"equivalent={equivalence_passed}",
            flush=True,
        )

    speedups = [
        comparison["reference_over_candidate_speedup"] for comparison in comparisons
    ]
    geometric_speedup = math.prod(speedups) ** (1 / len(speedups))
    equivalence_passed = all(item["equivalence_passed"] for item in comparisons)
    performance_passed = (
        geometric_speedup >= 1.05
        and min(speedups) >= 0.95
        and all(item["candidate_peak_extra_mib_delta"] <= 0 for item in comparisons)
    )
    report = {
        "run_dir": str(args.run_dir),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": "torch.bfloat16",
        "reference_backend": args.reference_backend,
        "candidate_backend": args.candidate_backend,
        "flex_block_size": args.flex_block_size,
        "attention_topology_input": (
            "compact_token_labels"
            if "flex_attention" in (args.reference_backend, args.candidate_backend)
            else "dense_boolean_mask"
        ),
        "method": (
            "same seeded epoch-0 real training batches; trainable saved LoRA and decision "
            "head; gradient checkpointing; autocast BF16; forward plus backward only; "
            "equivalence gated on probabilities, centered logits, loss, and gradients; "
            "raw logits retained as an informational diagnostic"
        ),
        "selection_report": (
            str(args.selection_report) if args.selection_report is not None else None
        ),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "thresholds": {
            "loss_atol": args.loss_atol,
            "loss_rtol": args.loss_rtol,
            "probability_atol": args.probability_atol,
            "centered_logit_atol": args.centered_logit_atol,
            "gradient_cosine_min": args.gradient_cosine_min,
            "gradient_relative_l2_max": args.gradient_relative_l2_max,
            "geometric_speedup_min": 1.05,
            "per_case_speedup_min": 0.95,
            "candidate_peak_memory_must_not_increase": True,
        },
        "selection": [
            {key: value for key, value in case.items() if key != "batch"} for case in cases
        ],
        "backends": backend_results,
        "comparisons": comparisons,
        "geometric_mean_speedup": geometric_speedup,
        "equivalence_passed": equivalence_passed,
        "performance_passed": performance_passed,
        "training_gate_passed": equivalence_passed and performance_passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"training benchmark saved: {output_path}\n"
        f"equivalence_passed={equivalence_passed} "
        f"performance_passed={performance_passed} "
        f"geometric_speedup={geometric_speedup:.2f}x",
        flush=True,
    )
    if not equivalence_passed:
        raise SystemExit(f"{args.candidate_backend} training equivalence gate failed")
    if not performance_passed and not args.allow_performance_failure:
        raise SystemExit(f"{args.candidate_backend} training performance gate failed")


if __name__ == "__main__":
    main()
