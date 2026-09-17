"""ZINC15 sampling core: stream *.src.txt, RDKit canonicalize + dedup, keep the lowest seeded blake2b hash ranks, write shards + manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import math
from multiprocessing import Pool
import time
from pathlib import Path
from typing import Iterator

from rdkit import RDLogger
from rdkit.Chem import AllChem

from pahelix.utils.compound_tools import get_largest_mol, split_rdkit_mol_obj

try:
    from src.sae_config import save_json, to_jsonable
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from .sae_config import save_json, to_jsonable


RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class Zinc15SamplingConfig:
    input_dir: Path
    output_dir: Path
    target_size: int = 2_000_000
    oversample_fraction: float = 0.10
    shard_size: int = 100_000
    seed: int = 42
    file_glob: str = "*.src.txt"
    preserve_stereo: bool = True
    keep_largest_fragment: bool = True
    max_rows: int | None = None
    max_files: int | None = None
    log_every_rows: int = 1_000_000
    num_workers: int = 1
    worker_chunksize: int = 256

    @property
    def candidate_size(self) -> int:
        return int(math.ceil(self.target_size * (1.0 + self.oversample_fraction)))

    def to_dict(self) -> dict[str, object]:
        return to_jsonable(self)


@dataclass
class SamplingStats:
    files_seen: int = 0
    rows_seen: int = 0
    blank_rows: int = 0
    invalid_smiles_rows: int = 0
    valid_rows: int = 0
    multi_fragment_rows: int = 0
    duplicate_candidates_seen: int = 0
    candidate_replacements: int = 0
    hash_collisions: int = 0
    candidate_size: int = 0
    selected_rows: int = 0
    shard_files_written: int = 0
    elapsed_sec: float = 0.0


def _iter_source_files(cfg: Zinc15SamplingConfig) -> list[Path]:
    files = sorted(cfg.input_dir.glob(cfg.file_glob))
    if cfg.max_files is not None:
        files = files[: cfg.max_files]
    if not files:
        raise FileNotFoundError(
            f"No input files matched {cfg.file_glob!r} under {cfg.input_dir}"
        )
    return files


def _iter_source_smiles(
    cfg: Zinc15SamplingConfig,
    stats: SamplingStats,
) -> Iterator[str]:
    rows_limit = cfg.max_rows
    for file_path in _iter_source_files(cfg):
        stats.files_seen += 1
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if rows_limit is not None and stats.rows_seen >= rows_limit:
                    return
                stats.rows_seen += 1
                stripped = line.strip()
                if not stripped:
                    stats.blank_rows += 1
                    continue
                token = stripped.split()[0]
                if not token:
                    stats.blank_rows += 1
                    continue
                yield token


def _canonicalize_worker(
    payload: tuple[str, bool, bool],
) -> tuple[str | None, bool, bool]:
    smiles, preserve_stereo, keep_largest_fragment = payload
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None, True, False

    multi_fragment = False
    if keep_largest_fragment:
        mol_species = split_rdkit_mol_obj(mol)
        if len(mol_species) > 1:
            multi_fragment = True
            mol = get_largest_mol(mol_species)
        elif len(mol_species) == 1:
            mol = mol_species[0]

    canonical = AllChem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=preserve_stereo,
    )
    if not canonical:
        return None, True, multi_fragment
    return canonical, False, multi_fragment


def _hash_rank(smiles: str, seed: int) -> int:
    person = f"z15-{seed}".encode("utf-8")[:16].ljust(16, b"\0")
    digest = hashlib.blake2b(
        smiles.encode("utf-8"),
        digest_size=16,
        person=person,
    ).digest()
    return int.from_bytes(digest, "big", signed=False)


def _peek_worst_candidate(
    heap: list[tuple[int, str]],
    selected: dict[str, int],
) -> tuple[int, str]:
    while heap:
        neg_rank, smiles = heap[0]
        rank = -neg_rank
        current_rank = selected.get(smiles)
        if current_rank == rank:
            return rank, smiles
        heapq.heappop(heap)
    raise RuntimeError("Candidate heap unexpectedly empty.")


def _pop_worst_candidate(
    heap: list[tuple[int, str]],
    selected: dict[str, int],
) -> tuple[int, str]:
    while heap:
        neg_rank, smiles = heapq.heappop(heap)
        rank = -neg_rank
        current_rank = selected.get(smiles)
        if current_rank == rank:
            return rank, smiles
    raise RuntimeError("Candidate heap unexpectedly empty.")


def _write_output_shards(
    candidates: list[tuple[int, str]],
    cfg: Zinc15SamplingConfig,
) -> int:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.output_dir / "shards.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for shard_index, start in enumerate(range(0, len(candidates), cfg.shard_size)):
            end = min(start + cfg.shard_size, len(candidates))
            shard_candidates = candidates[start:end]
            shard_path = cfg.output_dir / f"part-{shard_index:05d}"
            with shard_path.open("w", encoding="utf-8") as handle:
                for _, smiles in shard_candidates:
                    handle.write(smiles)
                    handle.write("\n")
            manifest.write(
                f'{{"file_name":"{shard_path.name}","num_rows":{len(shard_candidates)}}}\n'
            )
    return int(math.ceil(len(candidates) / cfg.shard_size))


def run_zinc15_sampling(cfg: Zinc15SamplingConfig) -> dict[str, object]:
    start_time = time.time()
    stats = SamplingStats(candidate_size=cfg.candidate_size)
    selected: dict[str, int] = {}
    candidate_heap: list[tuple[int, str]] = []

    source_files = _iter_source_files(cfg)
    source_manifest = cfg.output_dir / "source_files.txt"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text(
        "\n".join(file_path.name for file_path in source_files) + "\n",
        encoding="utf-8",
    )

    next_log_at = cfg.log_every_rows
    raw_iter = _iter_source_smiles(cfg, stats)
    worker_payloads = (
        (raw_smiles, cfg.preserve_stereo, cfg.keep_largest_fragment)
        for raw_smiles in raw_iter
    )
    if cfg.num_workers <= 1:
        canonical_iter = (
            _canonicalize_worker(payload)
            for payload in worker_payloads
        )
    else:
        pool = Pool(processes=cfg.num_workers)
        canonical_iter = pool.imap(
            _canonicalize_worker,
            worker_payloads,
            chunksize=cfg.worker_chunksize,
        )

    try:
        for canonical, invalid_flag, multi_fragment_flag in canonical_iter:
            if invalid_flag:
                stats.invalid_smiles_rows += 1
            else:
                stats.valid_rows += 1
            if multi_fragment_flag:
                stats.multi_fragment_rows += 1
            if canonical is None:
                continue

            candidate_rank = _hash_rank(canonical, cfg.seed)
            existing_rank = selected.get(canonical)
            if existing_rank is not None:
                if existing_rank != candidate_rank:
                    stats.hash_collisions += 1
                stats.duplicate_candidates_seen += 1
                continue

            if len(selected) < cfg.candidate_size:
                selected[canonical] = candidate_rank
                heapq.heappush(candidate_heap, (-candidate_rank, canonical))
            else:
                worst_rank, _ = _peek_worst_candidate(candidate_heap, selected)
                if candidate_rank >= worst_rank:
                    continue
                selected[canonical] = candidate_rank
                heapq.heappush(candidate_heap, (-candidate_rank, canonical))
                _, evicted_smiles = _pop_worst_candidate(candidate_heap, selected)
                del selected[evicted_smiles]
                stats.candidate_replacements += 1

            if cfg.log_every_rows and stats.rows_seen >= next_log_at:
                print(
                    f"[zinc15-sampler] rows={stats.rows_seen:,} "
                    f"valid={stats.valid_rows:,} selected={len(selected):,}"
                )
                next_log_at += cfg.log_every_rows
    finally:
        if cfg.num_workers > 1:
            pool.close()
            pool.join()

    ordered_candidates = sorted(
        ((rank, smiles) for smiles, rank in selected.items()),
        key=lambda item: (item[0], item[1]),
    )
    stats.selected_rows = len(ordered_candidates)
    stats.shard_files_written = _write_output_shards(ordered_candidates, cfg)
    stats.elapsed_sec = time.time() - start_time

    summary = {
        "config": cfg.to_dict(),
        "stats": to_jsonable(stats),
        "output_dir": str(cfg.output_dir),
        "source_manifest": str(source_manifest),
    }
    save_json(cfg.output_dir / "sampling_summary.json", summary)
    return summary
