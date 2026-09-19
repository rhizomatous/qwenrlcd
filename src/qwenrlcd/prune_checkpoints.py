from __future__ import annotations

import argparse
from pathlib import Path

from .checkpointing import (
    checkpoint_cleanup_candidates,
    prune_checkpoints,
    resolve_resume_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview or remove old training checkpoints")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--keep", required=True, type=int, dest="keep_last")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Delete the previewed directories")
    args = parser.parse_args()

    if args.keep_last > 0:
        resolve_resume_checkpoint("latest", args.run_dir)
    targets = checkpoint_cleanup_candidates(
        args.run_dir,
        keep_last=args.keep_last,
        include_incomplete=args.include_incomplete,
    )
    total_bytes = 0
    for path in targets:
        size = sum(
            child.stat().st_size
            for child in path.rglob("*")
            if child.is_file() and not child.is_symlink()
        )
        total_bytes += size
        print(f"would remove {path} ({size / 2**30:.2f} GiB)")
    print(f"reclaimable {total_bytes / 2**30:.2f} GiB")
    if args.apply:
        removed = prune_checkpoints(
            args.run_dir,
            keep_last=args.keep_last,
            include_incomplete=args.include_incomplete,
        )
        if removed != targets:
            raise RuntimeError("checkpoint targets changed while pruning")
        print(f"removed {len(removed)} checkpoint directories")


if __name__ == "__main__":
    main()
