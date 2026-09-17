#!/usr/bin/env python3
"""Deterministic ZINC15 sampler: canonicalize, dedup, seeded hash-rank selection. Paper: 2M + 10% oversample -> 2,199,971 molecules (seed 42)."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a reproducible ZINC15 corpus from catalogs/source with "
            "minimal cleaning and deterministic hash-rank selection."
        ),
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # 2_000_000 + 10% oversample → dedup/cleaning 후 ~2.2M. 바꾸면 3D 캐시 재생성 필요.
    parser.add_argument("--target-size", type=int, default=2_000_000)
    parser.add_argument("--oversample-fraction", type=float, default=0.10)
    parser.add_argument("--shard-size", type=int, default=100_000)
    # src.zinc15_sampling의 결정적 hash-rank 선택에 사용. seed를 바꾸면 코퍼스 전체가 바뀐다.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--file-glob", type=str, default="*.src.txt")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--log-every-rows", type=int, default=1_000_000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-chunksize", type=int, default=256)
    parser.add_argument("--drop-stereo", action="store_true")
    parser.add_argument("--keep-all-fragments", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from src.zinc15_sampling import Zinc15SamplingConfig, run_zinc15_sampling

    cfg = Zinc15SamplingConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
        oversample_fraction=args.oversample_fraction,
        shard_size=args.shard_size,
        seed=args.seed,
        file_glob=args.file_glob,
        preserve_stereo=not args.drop_stereo,
        keep_largest_fragment=not args.keep_all_fragments,
        max_rows=args.max_rows,
        max_files=args.max_files,
        log_every_rows=args.log_every_rows,
        num_workers=args.num_workers,
        worker_chunksize=args.worker_chunksize,
    )
    summary = run_zinc15_sampling(cfg)
    print(summary)


if __name__ == "__main__":
    main()
