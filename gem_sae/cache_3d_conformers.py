#!/usr/bin/env python3
"""Build MMFF3d conformer caches (InMemoryDataset NPZ) for a downstream dataset or a sharded SMILES corpus; resumable."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.featurizer import DownstreamTransformFn
from src.utils import get_dataset, get_downstream_task_names


def cache_dataset(
    dataset_name: str,
    data_path: str,
    cached_data_path: str,
    num_workers: int = 48,
) -> None:
    """Cache a downstream dataset (same as finetune --task=data)."""
    task_names = get_downstream_task_names(dataset_name, data_path)
    dataset = get_dataset(dataset_name, data_path, task_names)
    transform_fn = DownstreamTransformFn()
    dataset.transform(transform_fn, num_workers=num_workers, drop_none=True)
    dataset.save_data(cached_data_path)
    print(f"[cache_3d] {dataset_name}: saved {len(dataset)} molecules to {cached_data_path}")


def cache_smiles_dir(
    smiles_dir: str,
    cached_data_path: str,
    num_workers: int = 48,
) -> None:
    """Cache a plain SMILES directory (e.g. ZINC15 shards) with one shared worker pool."""
    import gc
    import multiprocessing as mp
    import time

    from pahelix.utils.data_utils import save_data_list_to_npz
    from src.sae_extraction import _is_smiles_file, _iter_smiles, _transform_smiles

    smiles_dir_path = Path(smiles_dir)
    cache_root = Path(cached_data_path)
    cache_root.mkdir(parents=True, exist_ok=True)

    if smiles_dir_path.is_dir():
        shard_files = sorted(
            p for p in smiles_dir_path.iterdir() if p.is_file() and _is_smiles_file(p)
        )
    else:
        shard_files = [smiles_dir_path]

    if not shard_files:
        raise ValueError(f"No SMILES shards found under {smiles_dir_path}")

    print(f"[cache_3d] caching {len(shard_files)} shards into {cache_root} "
          f"with num_workers={num_workers} (shared pool + imap_unordered streaming)",
          flush=True)

    # Part numbering continues after any pre-existing part files so resume is
    # gap-free and compatible with InMemoryDataset loaders.
    next_part_idx = len(list(cache_root.glob("part-*.npz")))
    total_saved = 0
    t_all = time.time()
    CHUNK_PER_PART = 10000

    def _cleanup_partial_parts(from_idx: int) -> int:
        """Delete part files with index >= from_idx left by a crashed shard."""
        removed = 0
        for p in cache_root.glob("part-*.npz"):
            try:
                idx = int(p.stem.split("-")[-1])
            except ValueError:
                continue
            if idx >= from_idx:
                p.unlink()
                removed += 1
        return removed

    pool = mp.Pool(processes=num_workers)
    try:
        for shard_idx, shard_path in enumerate(shard_files):
            done_marker = cache_root / f".shard-{shard_idx:04d}.done"
            start_marker = cache_root / f".shard-{shard_idx:04d}.start"
            if done_marker.exists():
                print(f"[cache_3d] shard {shard_idx:04d} ({shard_path.name}) "
                      f"already done, skip", flush=True)
                continue

            # Resume cleanup: ``.start`` without ``.done`` means the previous
            # attempt on this shard crashed mid-way. Delete every part file
            # produced after that attempt's starting index before retrying so
            # we never double-save.
            if start_marker.exists():
                crashed_start_idx = int(start_marker.read_text().strip())
                removed = _cleanup_partial_parts(crashed_start_idx)
                next_part_idx = len(list(cache_root.glob("part-*.npz")))
                print(f"[cache_3d] shard {shard_idx:04d}: recovered from crashed "
                      f"attempt (cleaned {removed} partial parts, "
                      f"next_part_idx={next_part_idx})", flush=True)
                start_marker.unlink()

            # Record the starting part index for this attempt so a later crash
            # recovery knows exactly which partial parts to delete.
            shard_start_idx = next_part_idx
            start_marker.write_text(str(shard_start_idx))

            t0 = time.time()
            smiles_list = _iter_smiles(shard_path)
            print(f"[cache_3d] shard {shard_idx:04d} ({shard_path.name}): "
                  f"{len(smiles_list)} SMILES, starting transform", flush=True)

            # Streaming: collect valid results in a buffer, flush to disk every
            # CHUNK_PER_PART molecules so memory stays O(CHUNK).
            buf: list = []
            n_failed = 0
            n_seen = 0
            for r in pool.imap_unordered(_transform_smiles, smiles_list, chunksize=256):
                n_seen += 1
                if r is None:
                    n_failed += 1
                else:
                    buf.append(r)
                    if len(buf) >= CHUNK_PER_PART:
                        fname = cache_root / f"part-{next_part_idx:06d}.npz"
                        save_data_list_to_npz(buf, str(fname))
                        next_part_idx += 1
                        buf = []
                if n_seen % 20000 == 0:
                    elapsed = time.time() - t0
                    rate = n_seen / elapsed if elapsed > 0 else 0.0
                    print(f"[cache_3d]   progress {n_seen}/{len(smiles_list)} "
                          f"valid={n_seen - n_failed} failed={n_failed} "
                          f"{rate:.0f} mol/s", flush=True)

            # flush tail
            if buf:
                fname = cache_root / f"part-{next_part_idx:06d}.npz"
                save_data_list_to_npz(buf, str(fname))
                next_part_idx += 1

            done_marker.touch()
            # Shard committed; the ``.start`` marker has served its purpose.
            if start_marker.exists():
                start_marker.unlink()
            shard_saved = n_seen - n_failed
            total_saved += shard_saved
            dt = time.time() - t0
            rate = len(smiles_list) / dt if dt > 0 else 0.0
            print(f"[cache_3d] shard {shard_idx:04d} done: "
                  f"{shard_saved}/{len(smiles_list)} valid, failed={n_failed}, "
                  f"{dt:.0f}s ({rate:.0f} mol/s). "
                  f"running total={total_saved}, elapsed={time.time() - t_all:.0f}s",
                  flush=True)
            gc.collect()
    finally:
        pool.close()
        pool.join()

    final_parts = sorted(cache_root.glob("part-*.npz"))
    print(f"[cache_3d] ALL SHARDS DONE: {total_saved} new mols saved, "
          f"{len(final_parts)} part-*.npz in {cache_root}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Cache MMFF3d conformers in InMemoryDataset NPZ format."
    )
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Downstream dataset name (bbbp, hiv, etc.)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to dataset directory")
    parser.add_argument("--smiles-dir", type=str, default=None,
                        help="Path to plain SMILES directory (alternative to --dataset-name)")
    parser.add_argument("--cached-data-path", type=str, required=True,
                        help="Output path for NPZ cache")
    parser.add_argument("--num-workers", type=int, default=48)
    args = parser.parse_args()

    if args.dataset_name and args.data_path:
        cache_dataset(args.dataset_name, args.data_path, args.cached_data_path, args.num_workers)
    elif args.smiles_dir:
        cache_smiles_dir(args.smiles_dir, args.cached_data_path, args.num_workers)
    else:
        parser.error("Provide either --dataset-name + --data-path, or --smiles-dir")


if __name__ == "__main__":
    main()
