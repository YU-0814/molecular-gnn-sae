"""Run molecules through the frozen GeoGNN encoder and write per-layer node/graph activations as chunk NPZ shards + manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import numpy as np
from rdkit.Chem import AllChem

import paddle
import pgl

from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_MMFF3d


def _transform_smiles(smiles: str) -> dict[str, Any] | None:
    """Standalone function for multiprocessing Pool (must be top-level picklable)."""
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        data = mol_to_geognn_graph_data_MMFF3d(mol)
    except Exception:
        return None
    if data is None:
        return None
    data["smiles"] = smiles
    return data

try:
    from src.sae_config import GemActivationExtractionConfig, save_json
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from .sae_config import GemActivationExtractionConfig, save_json


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


class GemInferenceCollateFn:
    def __init__(
        self,
        atom_names: list[str],
        bond_names: list[str],
        bond_float_names: list[str],
        bond_angle_float_names: list[str],
    ) -> None:
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names

    def _flat_shapes(self, features: dict[str, np.ndarray]) -> None:
        for name in features:
            features[name] = features[name].reshape([-1])

    def __call__(self, data_list: list[dict[str, Any]]) -> tuple[pgl.Graph, pgl.Graph, dict[str, Any]] | None:
        data_list = [data for data in data_list if data is not None]
        if not data_list:
            return None

        atom_bond_graph_list = []
        bond_angle_graph_list = []
        smiles = []
        node_counts = []
        for data in data_list:
            smiles.append(data["smiles"])
            node_counts.append(len(data[self.atom_names[0]]))
            atom_bond_graph_list.append(
                pgl.Graph(
                    num_nodes=len(data[self.atom_names[0]]),
                    edges=data["edges"],
                    node_feat={
                        name: data[name].reshape([-1, 1]) for name in self.atom_names
                    },
                    edge_feat={
                        name: data[name].reshape([-1, 1])
                        for name in self.bond_names + self.bond_float_names
                    },
                )
            )
            bond_angle_graph_list.append(
                pgl.Graph(
                    num_nodes=len(data["edges"]),
                    edges=data["BondAngleGraph_edges"],
                    node_feat={},
                    edge_feat={
                        name: data[name].reshape([-1, 1])
                        for name in self.bond_angle_float_names
                    },
                )
            )

        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        self._flat_shapes(bond_angle_graph.node_feat)
        self._flat_shapes(bond_angle_graph.edge_feat)
        return atom_bond_graph, bond_angle_graph, {
            "smiles": smiles,
            "node_counts": np.asarray(node_counts, dtype=np.int64),
        }


@dataclass
class _ChunkBuffer:
    output_dir: Path
    save_dtype: str
    chunk_index: int = 0
    d_in: int | None = None
    total_samples: int = 0
    total_graphs: int = 0

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_manifest_path = self.output_dir / "chunks.jsonl"
        # Clear stale files from previous runs to prevent data contamination
        if self.chunk_manifest_path.exists():
            self.chunk_manifest_path.unlink()
        for old_npz in self.output_dir.glob("chunk_*.npz"):
            old_npz.unlink()
        self._activations: list[np.ndarray] = []
        self._graph_ids: list[np.ndarray] = []
        self._buffer_graphs = 0

    def append(
        self,
        activations: np.ndarray,
        graph_ids: np.ndarray,
        *,
        num_graphs: int,
    ) -> None:
        if activations.ndim != 2:
            raise ValueError(f"Expected rank-2 activations, got shape {activations.shape}")
        if activations.shape[0] != graph_ids.shape[0]:
            raise ValueError("Activation rows and graph_ids length must match.")
        if self.d_in is None:
            self.d_in = int(activations.shape[1])
        self._activations.append(activations)
        self._graph_ids.append(graph_ids)
        self._buffer_graphs += num_graphs

    @property
    def buffered_graphs(self) -> int:
        return self._buffer_graphs

    def flush(self) -> None:
        if not self._activations:
            return
        activations = np.concatenate(self._activations, axis=0).astype(
            self.save_dtype, copy=False
        )
        graph_ids = np.concatenate(self._graph_ids, axis=0).astype(np.int64, copy=False)
        file_name = f"chunk_{self.chunk_index:06d}.npz"
        np.savez(self.output_dir / file_name, activations=activations, graph_ids=graph_ids)
        with self.chunk_manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "file_name": file_name,
                        "num_samples": int(activations.shape[0]),
                        "num_graphs": int(np.unique(graph_ids).shape[0]),
                    }
                )
                + "\n"
            )
        self.total_samples += int(activations.shape[0])
        self.total_graphs += int(np.unique(graph_ids).shape[0])
        self.chunk_index += 1
        self._activations = []
        self._graph_ids = []
        self._buffer_graphs = 0


_SMILES_TEXT_SUFFIXES = {".smi", ".csv", ".txt"}
_SMILES_META_FILENAMES = {
    "sampling_summary.json",
    "shards.jsonl",
    "source_files.txt",
}


def _is_smiles_file(path: Path) -> bool:
    """True for part-* shards and *.smi/*.txt/*.csv files."""
    if path.name in _SMILES_META_FILENAMES:
        return False
    if path.name.startswith("part-"):
        return True
    suffix = path.suffix.lower()
    return suffix in _SMILES_TEXT_SUFFIXES


def _iter_smiles(path: Path) -> list[str]:
    if path.is_file():
        if path.suffix.lower() == ".csv":
            import csv

            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                return [row["smiles"].strip() for row in reader if row.get("smiles")]
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    smiles: list[str] = []
    for file_path in sorted(p for p in path.iterdir() if p.is_file()):
        if not _is_smiles_file(file_path):
            continue
        smiles.extend(_iter_smiles(file_path))
    return smiles


def _is_cached_npz(path: Path) -> bool:
    """Check if path is an InMemoryDataset NPZ cache directory."""
    if not path.is_dir():
        return False
    return any(path.glob("part-*.npz"))


@paddle.no_grad()
def forward_with_capture(
    compound_encoder: GeoGNNModel,
    atom_bond_graph: pgl.Graph,
    bond_angle_graph: pgl.Graph,
    capture_layers: tuple[int, ...],
    *,
    save_graph_repr: bool,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    capture_set = set(capture_layers)
    node_hidden = compound_encoder.init_atom_embedding(atom_bond_graph.node_feat)
    bond_embed = compound_encoder.init_bond_embedding(atom_bond_graph.edge_feat)
    edge_hidden = bond_embed + compound_encoder.init_bond_float_rbf(atom_bond_graph.edge_feat)

    node_captures: dict[int, np.ndarray] = {}
    graph_captures: dict[int, np.ndarray] = {}
    if 0 in capture_set:
        node_captures[0] = _to_numpy(node_hidden)
        if save_graph_repr:
            graph_captures[0] = _to_numpy(
                compound_encoder.graph_pool(atom_bond_graph, node_hidden)
            )

    node_hidden_list = [node_hidden]
    edge_hidden_list = [edge_hidden]
    for layer_id in range(compound_encoder.layer_num):
        node_hidden = compound_encoder.atom_bond_block_list[layer_id](
            atom_bond_graph,
            node_hidden_list[layer_id],
            edge_hidden_list[layer_id],
        )

        cur_edge_hidden = compound_encoder.bond_embedding_list[layer_id](
            atom_bond_graph.edge_feat
        )
        cur_edge_hidden = cur_edge_hidden + compound_encoder.bond_float_rbf_list[layer_id](
            atom_bond_graph.edge_feat
        )
        cur_angle_hidden = compound_encoder.bond_angle_float_rbf_list[layer_id](
            bond_angle_graph.edge_feat
        )
        edge_hidden = compound_encoder.bond_angle_block_list[layer_id](
            bond_angle_graph,
            cur_edge_hidden,
            cur_angle_hidden,
        )

        node_hidden_list.append(node_hidden)
        edge_hidden_list.append(edge_hidden)

        layer_number = layer_id + 1
        if layer_number in capture_set:
            node_captures[layer_number] = _to_numpy(node_hidden)
            if save_graph_repr:
                graph_captures[layer_number] = _to_numpy(
                    compound_encoder.graph_pool(atom_bond_graph, node_hidden)
                )
    return node_captures, graph_captures


def _expand_graph_ids(graph_ids: list[int], node_counts: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(graph_ids, dtype=np.int64), node_counts.astype(np.int64))


def _create_writers(
    cfg: GemActivationExtractionConfig,
) -> dict[tuple[str, int], _ChunkBuffer]:
    writers: dict[tuple[str, int], _ChunkBuffer] = {}
    for layer in cfg.capture_layers:
        if cfg.representation in ("node", "both"):
            writers[("node", layer)] = _ChunkBuffer(
                output_dir=cfg.output_dir / "node" / f"layer_{layer:02d}",
                save_dtype=cfg.save_dtype,
            )
        if cfg.representation in ("graph", "both"):
            writers[("graph", layer)] = _ChunkBuffer(
                output_dir=cfg.output_dir / "graph" / f"layer_{layer:02d}",
                save_dtype=cfg.save_dtype,
            )
    return writers


def run_activation_extraction(cfg: GemActivationExtractionConfig) -> dict[str, Any]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    compound_encoder_config = load_json_config(str(cfg.compound_encoder_config))
    compound_encoder = GeoGNNModel(compound_encoder_config)
    if cfg.init_model is not None:
        compound_encoder.set_state_dict(paddle.load(str(cfg.init_model)))
    compound_encoder.eval()

    max_layer = compound_encoder_config["layer_num"]
    for layer in cfg.capture_layers:
        if layer < 0 or layer > max_layer:
            raise ValueError(
                f"capture_layer {layer} is out of range [0, {max_layer}] "
                f"for a {max_layer}-layer GeoGNN."
            )

    # --- Load data: from NPZ cache (fast) or raw SMILES (slow MMFF3d) ---
    if _is_cached_npz(cfg.data_path):
        print(f"[extract] Loading cached NPZ from {cfg.data_path} ...", flush=True)
        dataset = InMemoryDataset(npz_data_path=str(cfg.data_path))
        if cfg.max_molecules is not None:
            dataset = InMemoryDataset(data_list=dataset.data_list[:cfg.max_molecules])
        transformed = dataset.data_list
        print(f"[extract] Loaded {len(transformed)} cached molecules.", flush=True)
    else:
        smiles_list = _iter_smiles(cfg.data_path)
        if cfg.max_molecules is not None:
            smiles_list = smiles_list[: cfg.max_molecules]
        total_smiles = len(smiles_list)

        num_workers = cfg.num_workers
        print(
            f"[extract] Transforming {total_smiles} molecules with {num_workers} workers ...",
            flush=True,
        )
        transformed: list[dict[str, Any]] = []
        done = 0
        failed = 0
        t0 = time.time()
        with mp.Pool(processes=num_workers) as pool:
            # Ordered imap keeps SMILES and activation rows aligned; do not use imap_unordered.
            for result in pool.imap(
                _transform_smiles, smiles_list, chunksize=256
            ):
                done += 1
                if result is not None:
                    transformed.append(result)
                else:
                    failed += 1
                if done % 10000 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (total_smiles - done) / rate if rate > 0 else 0
                    print(
                        f"[extract] {done}/{total_smiles} "
                        f"({done/total_smiles*100:.1f}%) "
                        f"valid={len(transformed)} failed={failed} "
                        f"{rate:.0f} mol/s "
                        f"ETA {eta/60:.0f}min",
                        flush=True,
                    )
        elapsed = time.time() - t0
        print(
            f"[extract] Transform done: {len(transformed)}/{total_smiles} valid, "
            f"{failed} failed, in {elapsed:.0f}s ({total_smiles/elapsed:.0f} mol/s)",
            flush=True,
        )
    dataset = InMemoryDataset(data_list=transformed)

    collate_fn = GemInferenceCollateFn(
        atom_names=compound_encoder_config["atom_names"],
        bond_names=compound_encoder_config["bond_names"],
        bond_float_names=compound_encoder_config["bond_float_names"],
        bond_angle_float_names=compound_encoder_config["bond_angle_float_names"],
    )
    data_loader = dataset.get_data_loader(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    writers = _create_writers(cfg)
    graph_manifest_path = cfg.output_dir / "graphs.jsonl"
    if graph_manifest_path.exists():
        graph_manifest_path.unlink()

    total_graphs = 0
    invalid_batches = 0
    for batch in data_loader:
        if batch is None:
            invalid_batches += 1
            continue
        atom_bond_graph, bond_angle_graph, meta = batch
        atom_bond_graph = atom_bond_graph.tensor()
        bond_angle_graph = bond_angle_graph.tensor()

        node_captures, graph_captures = forward_with_capture(
            compound_encoder,
            atom_bond_graph,
            bond_angle_graph,
            cfg.capture_layers,
            save_graph_repr=cfg.representation in ("graph", "both"),
        )

        node_counts = meta["node_counts"]
        batch_graph_ids = list(range(total_graphs, total_graphs + len(node_counts)))
        total_graphs += len(batch_graph_ids)
        node_graph_ids = _expand_graph_ids(batch_graph_ids, node_counts)

        with graph_manifest_path.open("a", encoding="utf-8") as handle:
            for graph_id, num_nodes, smiles in zip(
                batch_graph_ids, node_counts.tolist(), meta["smiles"], strict=True
            ):
                payload = {"graph_id": graph_id, "num_nodes": int(num_nodes)}
                if cfg.save_smiles_manifest:
                    payload["smiles"] = smiles
                handle.write(json.dumps(payload) + "\n")

        for layer, node_values in node_captures.items():
            if ("node", layer) in writers:
                writers[("node", layer)].append(
                    node_values,
                    node_graph_ids,
                    num_graphs=len(batch_graph_ids),
                )
            if ("graph", layer) in writers:
                graph_values = graph_captures[layer]
                writers[("graph", layer)].append(
                    graph_values,
                    np.asarray(batch_graph_ids, dtype=np.int64),
                    num_graphs=len(batch_graph_ids),
                )

        for writer in writers.values():
            if writer.buffered_graphs >= cfg.chunk_size_graphs:
                writer.flush()

    for writer in writers.values():
        writer.flush()

    if total_graphs == 0:
        raise RuntimeError(
            "No valid molecules were processed during activation extraction. "
            "Check the input dataset path and RDKit/MMFF preprocessing."
        )

    summary: dict[str, Any] = {
        "run_name": cfg.run_name,
        "data_path": str(cfg.data_path),
        "capture_layers": list(cfg.capture_layers),
        "representation": cfg.representation,
        "total_graphs": total_graphs,
        "invalid_batches": invalid_batches,
    }
    for (representation, layer), writer in writers.items():
        meta_path = writer.output_dir / "meta.json"
        save_json(
            meta_path,
            {
                "representation": representation,
                "layer": layer,
                "d_in": writer.d_in,
                "num_samples": writer.total_samples,
                "num_graphs": writer.total_graphs,
                "dtype": cfg.save_dtype,
            },
        )
        summary[f"{representation}_layer_{layer:02d}_samples"] = writer.total_samples
        summary[f"{representation}_layer_{layer:02d}_graphs"] = writer.total_graphs

    save_json(cfg.output_dir / "extraction_config.json", cfg.to_dict())
    save_json(cfg.output_dir / "extraction_summary.json", summary)
    return summary
