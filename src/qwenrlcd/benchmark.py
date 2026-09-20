from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from .formatting import render_state
from .schema import DecisionBundle, Option, Question, QuestionType

EMOTIONS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
)
STATE_SENTENCE = (
    "The customer wrote that the replacement arrived today, but the original "
    "charge still appears on the account."
)
MODEL_INPUT_KEYS = (
    "input_ids", "position_ids", "tree_attention_mask", "decision_indices"
)


def parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        numbers = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("expected comma-separated positive integers") from exc
    if not numbers or any(number < 1 for number in numbers) or len(set(numbers)) != len(numbers):
        raise ValueError("expected distinct positive integers")
    return numbers


def benchmark_state(tokenizer: Any, minimum_tokens: int) -> tuple[str, int]:
    if minimum_tokens < 1:
        raise ValueError("minimum_tokens must be positive")
    state = STATE_SENTENCE
    while True:
        example = DecisionBundle("length-check", state, (benchmark_question(0),))
        length = len(tokenizer(render_state(example), add_special_tokens=False)["input_ids"])
        if length >= minimum_tokens:
            return state, length
        state += " " + STATE_SENTENCE


def benchmark_question(index: int) -> Question:
    emotion = EMOTIONS[index]
    return Question(
        id=emotion,
        type=QuestionType.NOUL,
        instructions=f"Does this comment express {emotion}?",
        options=(Option("false"), Option("true")),
    )


def benchmark_bundles(
    state: str, question_count: int
) -> tuple[DecisionBundle, list[DecisionBundle]]:
    if not 1 <= question_count <= len(EMOTIONS):
        raise ValueError(f"question_count must be between 1 and {len(EMOTIONS)}")
    questions = tuple(benchmark_question(index) for index in range(question_count))
    bundled = DecisionBundle("benchmark-bundled", state, questions)
    singletons = [
        DecisionBundle(f"benchmark-{question.id}", state, (question,))
        for question in questions
    ]
    return bundled, singletons


def timing_summary(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms or any(value <= 0 for value in samples_ms):
        raise ValueError("timings must be positive and nonempty")
    ordered = sorted(samples_ms)
    return {
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def probabilities(logit_rows: list[list[float]]) -> list[list[float]]:
    result = []
    for row in logit_rows:
        if not row:
            raise ValueError("logit row cannot be empty")
        maximum = max(row)
        shifted = [math.exp(value - maximum) for value in row]
        total = sum(shifted)
        result.append([value / total for value in shifted])
    return result


def max_probability_delta(
    reference: list[list[float]], candidate: list[list[float]]
) -> float:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("probability rows must have equal nonzero length")
    if any(len(first) != len(second) for first, second in zip(reference, candidate, strict=True)):
        raise ValueError("probability rows must have matching choices")
    return max(
        abs(first - second)
        for reference_row, candidate_row in zip(reference, candidate, strict=True)
        for first, second in zip(reference_row, candidate_row, strict=True)
    )


def _cpu_batches(
    bundled: DecisionBundle, singletons: list[DecisionBundle],
    *, tokenizer: Any, config: dict[str, Any], singleton_batch_size: int,
) -> dict[str, list[dict[str, Any]]]:
    from .data import DecisionCollator, DecisionDataset

    def packed(bundles: list[DecisionBundle]) -> list[dict[str, Any]]:
        dataset = DecisionDataset(
            bundles, tokenizer,
            max_length=int(config["max_length"]),
            max_choices=int(config["max_choices"]),
            max_questions=int(config["max_questions"]),
            shuffle=False,
            seed=int(config["seed"]),
            require_targets=False,
            require_no_state_truncation=True,
        )
        return [dataset[index] for index in range(len(dataset))]

    collator = DecisionCollator(
        tokenizer,
        max_choices=int(config["max_choices"]),
        max_questions=int(config["max_questions"]),
    )
    bundled_packed = packed([bundled])
    singleton_packed = packed(singletons)
    return {
        "bundled": [collator(bundled_packed)],
        "sequential_singletons": [collator([item]) for item in singleton_packed],
        "batched_singletons": [
            collator(singleton_packed[start : start + singleton_batch_size])
            for start in range(0, len(singleton_packed), singleton_batch_size)
        ],
    }


def _benchmark_mode(
    model: Any, cpu_batches: list[dict[str, Any]], *,
    torch: Any, warmups: int, repeats: int, model_bytes: int,
) -> tuple[dict[str, Any], list[list[float]]]:
    batches = [
        {key: batch[key].to("cuda") for key in MODEL_INPUT_KEYS}
        for batch in cpu_batches
    ]
    attention_positions = sum(
        batch["input_ids"].shape[0] * batch["input_ids"].shape[1] ** 2
        for batch in cpu_batches
    )
    padded_tokens = sum(batch["input_ids"].numel() for batch in cpu_batches)

    def forward_all() -> list[Any]:
        return [model(**batch) for batch in batches]

    timings_ms = []
    first_logits: list[list[float]] | None = None
    with torch.inference_mode():
        for _ in range(warmups):
            outputs = forward_all()
            del outputs
        torch.cuda.synchronize()
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = forward_all()
            torch.cuda.synchronize()
            timings_ms.append(1000 * (time.perf_counter() - start))
            if first_logits is None:
                first_logits = torch.cat(
                    [output.float().reshape(-1, output.shape[-1]) for output in outputs]
                ).cpu().tolist()
            del outputs
    del batches
    # Timing keeps all inputs resident so transfer is excluded. Measure peak
    # separately with only one forward's input resident at a time; otherwise
    # the sequential baseline would misleadingly retain every singleton mask.
    peak_bytes = model_bytes
    with torch.inference_mode():
        for cpu_batch in cpu_batches:
            batch = {key: cpu_batch[key].to("cuda") for key in MODEL_INPUT_KEYS}
            torch.cuda.reset_peak_memory_stats()
            output = model(**batch)
            torch.cuda.synchronize()
            peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated())
            del output, batch
    gc.collect()
    torch.cuda.empty_cache()
    return {
        **timing_summary(timings_ms),
        "samples_ms": timings_ms,
        "peak_allocated_mib": peak_bytes / 2**20,
        "peak_above_loaded_model_mib": max(0, peak_bytes - model_bytes) / 2**20,
        "padded_tokens": padded_tokens,
        "dense_attention_positions": attention_positions,
        "forward_calls": len(cpu_batches),
    }, first_logits or []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one tree-masked bundle against singleton forwards"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--question-counts", default="1,4,8,16,28")
    parser.add_argument("--state-tokens", default="64,1024")
    parser.add_argument("--singleton-batch-size", type=int, default=28)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    try:
        question_counts = parse_positive_ints(args.question_counts)
        state_tokens = parse_positive_ints(args.state_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    if max(question_counts) > len(EMOTIONS):
        parser.error(f"question counts cannot exceed {len(EMOTIONS)}")
    if min(args.singleton_batch_size, args.repeats) < 1 or args.warmups < 0:
        parser.error("batch size and repeats must be positive; warmups cannot be negative")

    output_path = args.output or args.run_dir / "parallel_speed_benchmark.json"
    if output_path.exists():
        parser.error(f"refusing to overwrite existing benchmark: {output_path}")
    if not (args.run_dir / "final" / "decision_head.pt").is_file():
        parser.error(f"no saved decision model in {args.run_dir / 'final'}")

    import torch
    from transformers import AutoTokenizer

    from .model import DecisionModel
    from .train import load_config

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required for this latency benchmark")
    config = load_config(args.run_dir / "training_config.json")
    if max(question_counts) > int(config["max_questions"]):
        parser.error("question count exceeds saved model configuration")
    tokenizer_dir = args.run_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir if tokenizer_dir.is_dir() else config["model_id"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = DecisionModel.load_components(
        args.run_dir / "final",
        model_id=config["model_id"],
        dtype=dtype,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        attn_implementation=config.get("attn_implementation", "eager"),
    ).to("cuda")
    model.eval()
    torch.cuda.synchronize()
    model_bytes = torch.cuda.memory_allocated()
    report: dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": str(dtype),
        "method": (
            "pretokenized GPU-resident inputs for synchronous model-only latency; "
            "peak memory measured with one forward's input resident at a time"
        ),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "singleton_batch_size": args.singleton_batch_size,
        "loaded_model_mib": model_bytes / 2**20,
        "cases": [],
    }
    print(
        f"GPU={report['gpu']} dtype={report['dtype']} repeats={args.repeats} "
        f"warmups={args.warmups} singleton_batch_size={args.singleton_batch_size}",
        flush=True,
    )

    for target_state_tokens in state_tokens:
        state, actual_state_tokens = benchmark_state(tokenizer, target_state_tokens)
        for question_count in question_counts:
            bundled, singletons = benchmark_bundles(state, question_count)
            cpu_batches = _cpu_batches(
                bundled, singletons,
                tokenizer=tokenizer, config=config,
                singleton_batch_size=args.singleton_batch_size,
            )
            modes: dict[str, Any] = {}
            predictions: dict[str, list[list[float]]] = {}
            for name, batches in cpu_batches.items():
                modes[name], logits = _benchmark_mode(
                    model, batches, torch=torch,
                    warmups=args.warmups, repeats=args.repeats, model_bytes=model_bytes,
                )
                predictions[name] = probabilities(logits)
                if len(predictions[name]) != question_count:
                    raise ValueError(f"{name} returned the wrong number of questions")
            bundle_ms = modes["bundled"]["p50_ms"]
            case = {
                "target_state_tokens": target_state_tokens,
                "actual_state_tokens": actual_state_tokens,
                "questions": question_count,
                "modes": modes,
                "sequential_over_bundle_speedup": modes["sequential_singletons"]["p50_ms"]
                / bundle_ms,
                "batched_over_bundle_speedup": modes["batched_singletons"]["p50_ms"]
                / bundle_ms,
                "max_probability_delta_vs_sequential": max_probability_delta(
                    predictions["bundled"], predictions["sequential_singletons"]
                ),
                "max_probability_delta_vs_batched": max_probability_delta(
                    predictions["bundled"], predictions["batched_singletons"]
                ),
            }
            report["cases"].append(case)
            max_delta = max(
                case["max_probability_delta_vs_sequential"],
                case["max_probability_delta_vs_batched"],
            )
            print(
                f"state={actual_state_tokens} questions={question_count} "
                f"bundle_p50={bundle_ms:.1f}ms "
                f"sequential_p50={modes['sequential_singletons']['p50_ms']:.1f}ms "
                f"batched_p50={modes['batched_singletons']['p50_ms']:.1f}ms "
                f"speedup_seq={case['sequential_over_bundle_speedup']:.2f}x "
                f"speedup_batch={case['batched_over_bundle_speedup']:.2f}x "
                f"max_probability_delta={max_delta:.5f}",
                flush=True,
            )
            print(
                "  peak_extra_mib "
                + " ".join(
                    f"{name}={modes[name]['peak_above_loaded_model_mib']:.1f}"
                    for name in modes
                ),
                flush=True,
            )
            print(
                "  dense_pairs_m "
                + " ".join(
                    f"{name}={modes[name]['dense_attention_positions'] / 1e6:.2f}"
                    for name in modes
                ),
                flush=True,
            )
            del cpu_batches
            gc.collect()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"benchmark saved: {output_path}")


if __name__ == "__main__":
    main()
