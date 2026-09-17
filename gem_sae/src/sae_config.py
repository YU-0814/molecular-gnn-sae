"""Config dataclasses for extraction (GemActivationExtractionConfig) and SAE training (GemSaeTrainConfig)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Any, Literal


def parse_layers_arg(raw: str) -> tuple[int, ...]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("At least one layer must be provided.")
    return tuple(values)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


@dataclass(frozen=True)
class GemActivationExtractionConfig:
    data_path: Path
    output_dir: Path
    compound_encoder_config: Path
    init_model: Path | None = None
    capture_layers: tuple[int, ...] = (8,)
    representation: Literal["node", "graph", "both"] = "both"
    batch_size: int = 64
    num_workers: int = 4
    chunk_size_graphs: int = 20_000
    save_dtype: Literal["float16", "float32"] = "float16"
    max_molecules: int | None = None
    save_smiles_manifest: bool = False
    run_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class GemSaeTrainConfig:
    activation_root: Path
    output_dir: Path
    layer: int
    representation: Literal["node", "graph"] = "graph"
    saelens_root: Path | None = None
    d_sae: int | None = None
    expansion_factor: int = 32  # paper: 32 -> 1024
    total_training_samples: int | None = None
    train_batch_size_samples: int = 4_096
    validation_fraction: float = 0.02
    holdout_eval_batches: int = 8
    lr: float = 3e-4
    lr_end: float | None = None
    lr_scheduler_name: Literal[
        "constant", "cosineannealing", "cosineannealingwarmrestarts"
    ] = "constant"
    lr_warm_up_steps: int = 0
    lr_decay_steps: int = 0
    n_restart_cycles: int = 1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    dead_feature_window: int = 1_000
    feature_sampling_window: int = 2_000
    n_batches_for_norm_estimate: int = 1_000
    n_checkpoints: int = 1
    save_final_checkpoint: bool = True
    checkpoint_path: Path | None = None
    normalize_activations: Literal[
        "none", "expected_average_only_in", "layer_norm"
    ] = "expected_average_only_in"
    jumprelu_sparsity_loss_mode: Literal["step", "tanh"] = "tanh"
    jumprelu_tanh_scale: float = 4.0
    jumprelu_bandwidth: float = 0.05
    jumprelu_init_threshold: float = 0.1
    jumprelu_l0_coefficient: float = 0.001
    jumprelu_l0_warm_up_steps: int = 0
    jumprelu_pre_act_loss_coefficient: float | None = 3e-6
    device: str = "cuda"
    autocast: bool = True
    seed: int = 42
    log_to_wandb: bool = True
    wandb_project: str = "gem_sae"
    wandb_entity: str | None = None
    wandb_id: str | None = None
    wandb_group: str | None = None
    run_name: str | None = None
    wandb_log_frequency: int = 10
    eval_every_n_wandb_logs: int = 20
    log_weights_to_wandb: bool = True
    log_optimizer_state_to_wandb: bool = False

    def resolved_d_sae(self, d_in: int) -> int:
        return self.d_sae if self.d_sae is not None else d_in * self.expansion_factor

    def resolved_lr_end(self) -> float:
        return self.lr_end if self.lr_end is not None else self.lr / 10.0

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)
