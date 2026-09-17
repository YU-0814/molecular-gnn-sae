#!/usr/bin/env python3
"""Extract GEM layer activations (node/graph) to chunk_*.npz. Paper: pretrained GEM, layer 8, ZINC15 2.2M and BBBP/ClinTox/BACE."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract GEM intermediate representations for SAELens training.",
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compound-encoder-config",
        type=Path,
        default=Path("model_configs/geognn_l8.json"),
    )
    # GEM checkpoint; required (paper: pretrain_models-chemrl_gem/class.pdparams).
    parser.add_argument("--init-model", type=Path, default=None)
    # The paper uses layer 8 only.
    parser.add_argument("--capture-layers", type=str, default="8")
    parser.add_argument(
        "--representation",
        choices=("node", "graph", "both"),
        default="both",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size-graphs", type=int, default=20000)
    # float16 halves disk usage; sae_data casts back to float32 for training.
    parser.add_argument("--save-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--save-smiles-manifest", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from src.sae_config import GemActivationExtractionConfig, parse_layers_arg
    from src.sae_extraction import run_activation_extraction

    cfg = GemActivationExtractionConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        compound_encoder_config=args.compound_encoder_config,
        init_model=args.init_model,
        capture_layers=parse_layers_arg(args.capture_layers),
        representation=args.representation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chunk_size_graphs=args.chunk_size_graphs,
        save_dtype=args.save_dtype,
        max_molecules=args.max_molecules,
        save_smiles_manifest=args.save_smiles_manifest,
        run_name=args.run_name,
    )
    summary = run_activation_extraction(cfg)
    print(summary)


if __name__ == "__main__":
    main()
