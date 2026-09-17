"""Chunked activation I/O: streaming data provider for SAELens and a held-out MSE/L0/EV evaluator."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ActivationChunkInfo:
    path: Path
    num_samples: int
    num_graphs: int


@dataclass(frozen=True)
class ActivationLayerMetadata:
    representation: str
    layer: int
    d_in: int
    num_samples: int
    num_graphs: int
    dtype: str


def _layer_dir(activation_root: Path, representation: str, layer: int) -> Path:
    return activation_root / representation / f"layer_{layer:02d}"


def load_layer_metadata(
    activation_root: Path, representation: str, layer: int
) -> ActivationLayerMetadata:
    layer_dir = _layer_dir(activation_root, representation, layer)
    payload = json.loads((layer_dir / "meta.json").read_text(encoding="utf-8"))
    return ActivationLayerMetadata(
        representation=payload["representation"],
        layer=int(payload["layer"]),
        d_in=int(payload["d_in"]),
        num_samples=int(payload["num_samples"]),
        num_graphs=int(payload["num_graphs"]),
        dtype=str(payload["dtype"]),
    )


def list_chunk_infos(
    activation_root: Path, representation: str, layer: int
) -> list[ActivationChunkInfo]:
    layer_dir = _layer_dir(activation_root, representation, layer)
    manifest = layer_dir / "chunks.jsonl"
    if manifest.exists():
        entries = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            entries.append(
                ActivationChunkInfo(
                    path=layer_dir / payload["file_name"],
                    num_samples=int(payload["num_samples"]),
                    num_graphs=int(payload["num_graphs"]),
                )
            )
        if entries:
            return entries

    paths = sorted(layer_dir.glob("chunk_*.npz"))
    entries: list[ActivationChunkInfo] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            num_samples = int(payload["activations"].shape[0])
            graph_ids = payload["graph_ids"]
            num_graphs = int(np.unique(graph_ids).shape[0])
        entries.append(
            ActivationChunkInfo(
                path=path,
                num_samples=num_samples,
                num_graphs=num_graphs,
            )
        )
    return entries


def split_chunk_infos(
    chunk_infos: list[ActivationChunkInfo],
    validation_fraction: float,
    *,
    min_validation_chunks: int = 1,
) -> tuple[list[ActivationChunkInfo], list[ActivationChunkInfo]]:
    if not chunk_infos:
        raise ValueError("No activation chunks were found.")
    if len(chunk_infos) == 1 or validation_fraction <= 0:
        return chunk_infos, []
    n_val = max(min_validation_chunks, int(round(len(chunk_infos) * validation_fraction)))
    n_val = min(len(chunk_infos) - 1, n_val)
    return chunk_infos[:-n_val], chunk_infos[-n_val:]


class NpzActivationDataProvider(Iterator[torch.Tensor]):
    def __init__(
        self,
        chunk_infos: list[ActivationChunkInfo],
        batch_size: int,
        *,
        seed: int = 42,
        shuffle_chunks: bool = True,
        shuffle_rows: bool = True,
    ) -> None:
        if not chunk_infos:
            raise ValueError("chunk_infos must not be empty.")
        self.chunk_infos = list(chunk_infos)
        self.batch_size = batch_size
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_rows = shuffle_rows
        self.generator = torch.Generator().manual_seed(seed)
        self._epoch_chunk_infos: list[ActivationChunkInfo] = []
        self._chunk_cursor = 0
        self._current_chunk: torch.Tensor | None = None
        self._row_cursor = 0
        self._reset_epoch()

    def _reset_epoch(self) -> None:
        self._epoch_chunk_infos = list(self.chunk_infos)
        if self.shuffle_chunks and len(self._epoch_chunk_infos) > 1:
            order = torch.randperm(
                len(self._epoch_chunk_infos), generator=self.generator
            ).tolist()
            self._epoch_chunk_infos = [self._epoch_chunk_infos[i] for i in order]
        self._chunk_cursor = 0
        self._current_chunk = None
        self._row_cursor = 0

    def _load_next_chunk(self) -> None:
        if self._chunk_cursor >= len(self._epoch_chunk_infos):
            self._reset_epoch()
        chunk_info = self._epoch_chunk_infos[self._chunk_cursor]
        self._chunk_cursor += 1
        with np.load(chunk_info.path, allow_pickle=False) as payload:
            activations = payload["activations"].astype(np.float32, copy=False)
        tensor = torch.from_numpy(activations)
        if self.shuffle_rows and tensor.shape[0] > 1:
            perm = torch.randperm(tensor.shape[0], generator=self.generator)
            tensor = tensor[perm]
        self._current_chunk = tensor
        self._row_cursor = 0

    def __iter__(self) -> "NpzActivationDataProvider":
        return self

    def __next__(self) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        filled = 0
        while filled < self.batch_size:
            if self._current_chunk is None or self._row_cursor >= self._current_chunk.shape[0]:
                self._load_next_chunk()
            assert self._current_chunk is not None
            remaining = self.batch_size - filled
            take_end = min(self._row_cursor + remaining, self._current_chunk.shape[0])
            part = self._current_chunk[self._row_cursor:take_end]
            self._row_cursor = take_end
            if part.numel() == 0:
                continue
            parts.append(part)
            filled += part.shape[0]
        return torch.cat(parts, dim=0)


class ChunkedActivationEvaluator:
    def __init__(
        self,
        chunk_infos: list[ActivationChunkInfo],
        batch_size: int,
        *,
        max_batches: int = 8,
        seed: int = 1234,
    ) -> None:
        self.chunk_infos = list(chunk_infos)
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.seed = seed

    @torch.no_grad()
    def __call__(
        self,
        sae: Any,
        _train_provider: Iterator[torch.Tensor],
        activation_scaler: Any,
    ) -> dict[str, Any]:
        if not self.chunk_infos:
            return {}

        provider = NpzActivationDataProvider(
            self.chunk_infos,
            batch_size=self.batch_size,
            seed=self.seed,
            shuffle_chunks=True,
            shuffle_rows=True,
        )

        mse_values: list[float] = []
        l0_values: list[float] = []
        explained_variance_values: list[float] = []
        for _ in range(self.max_batches):
            batch = next(provider).to(sae.device, non_blocking=True)
            scaled_batch = activation_scaler(batch)
            feature_acts, _hidden_pre = sae.encode_with_hidden_pre(scaled_batch)
            sae_out = sae.decode(feature_acts)
            per_item_l2 = (sae_out - scaled_batch).pow(2).sum(dim=-1)
            total_variance = (scaled_batch - scaled_batch.mean(0)).pow(2).sum(dim=-1)
            explained_variance = 1 - per_item_l2.mean() / total_variance.mean().clamp(
                min=1e-8
            )
            l0 = feature_acts.ne(0).float().sum(dim=-1).mean()
            mse_values.append(float(per_item_l2.mean().item()))
            l0_values.append(float(l0.item()))
            explained_variance_values.append(float(explained_variance.item()))

        return {
            "eval/losses/mse_loss": float(np.mean(mse_values)),
            "eval/metrics/l0": float(np.mean(l0_values)),
            "eval/metrics/explained_variance": float(
                np.mean(explained_variance_values)
            ),
        }
