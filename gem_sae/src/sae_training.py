"""SAE training entry point: seed, load chunks, build SAELens JumpReLU config, fit, save inference-form SAE, log to W&B."""

from __future__ import annotations

import random
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

try:
    from src.sae_config import GemSaeTrainConfig, save_json
    from src.sae_data import (
        ChunkedActivationEvaluator,
        NpzActivationDataProvider,
        list_chunk_infos,
        load_layer_metadata,
        split_chunk_infos,
    )
    from src.sae_wandb import (
        finish_wandb_run,
        log_wandb_artifact,
        save_wandb_metadata,
        start_wandb_run,
    )
except ModuleNotFoundError:
    from .sae_config import GemSaeTrainConfig, save_json
    from .sae_data import (
        ChunkedActivationEvaluator,
        NpzActivationDataProvider,
        list_chunk_infos,
        load_layer_metadata,
        split_chunk_infos,
    )
    from .sae_wandb import (
        finish_wandb_run,
        log_wandb_artifact,
        save_wandb_metadata,
        start_wandb_run,
    )


def ensure_saelens_importable(root: Path | None = None) -> None:
    """Prefer a SAELens source checkout at ``root`` if given; otherwise use the installed package."""
    if root is not None and str(root.resolve()) not in sys.path:
        sys.path.insert(0, str(root.resolve()))


def run_gem_sae_training(cfg: GemSaeTrainConfig) -> dict[str, Any]:
    ensure_saelens_importable(cfg.saelens_root)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    from sae_lens.config import LoggingConfig, SAETrainerConfig
    from sae_lens.saes.jumprelu_sae import JumpReLUTrainingSAEConfig
    from sae_lens.saes.sae import TrainingSAE
    from sae_lens.training.sae_trainer import SAETrainer

    layer_meta = load_layer_metadata(cfg.activation_root, cfg.representation, cfg.layer)
    chunk_infos = list_chunk_infos(cfg.activation_root, cfg.representation, cfg.layer)
    train_chunks, val_chunks = split_chunk_infos(chunk_infos, cfg.validation_fraction)
    available_training_samples = sum(chunk.num_samples for chunk in train_chunks)
    total_training_samples = cfg.total_training_samples or available_training_samples
    if total_training_samples <= 0:
        raise ValueError("total_training_samples must be positive.")
    if cfg.validation_fraction > 0 and not val_chunks:
        raise ValueError(
            f"validation_fraction={cfg.validation_fraction} requested but no validation "
            f"chunks were produced from {len(chunk_infos)} chunks. Extract smaller chunks "
            "or provide an explicit validation split."
        )
    if total_training_samples > available_training_samples and not val_chunks:
        raise ValueError(
            f"Refusing repeated training over a single activation chunk without validation: "
            f"requested {total_training_samples} samples but only {available_training_samples} "
            "are available and no holdout exists."
        )

    d_sae = cfg.resolved_d_sae(layer_meta.d_in)
    total_training_steps = total_training_samples // cfg.train_batch_size_samples
    if total_training_steps <= 0:
        raise ValueError(
            "train_batch_size_samples is larger than total_training_samples."
        )

    run_name = cfg.run_name or f"gem-{cfg.representation}-L{cfg.layer:02d}-jumprelu"
    wandb_group = cfg.wandb_group or f"gem-{cfg.representation}-L{cfg.layer:02d}"
    run_root = cfg.output_dir / cfg.representation / f"layer_{cfg.layer:02d}" / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    sae_cfg = JumpReLUTrainingSAEConfig(
        d_in=layer_meta.d_in,
        d_sae=d_sae,
        dtype="float32",
        device=cfg.device,
        apply_b_dec_to_input=True,
        normalize_activations=cfg.normalize_activations,
        decoder_init_norm=0.1,
        jumprelu_init_threshold=cfg.jumprelu_init_threshold,
        jumprelu_bandwidth=cfg.jumprelu_bandwidth,
        jumprelu_sparsity_loss_mode=cfg.jumprelu_sparsity_loss_mode,
        jumprelu_tanh_scale=cfg.jumprelu_tanh_scale,
        l0_coefficient=cfg.jumprelu_l0_coefficient,
        l0_warm_up_steps=cfg.jumprelu_l0_warm_up_steps if cfg.jumprelu_l0_warm_up_steps > 0 else 0,
        pre_act_loss_coefficient=cfg.jumprelu_pre_act_loss_coefficient,
    )
    sae_cfg.metadata.model_name = "GeoGNN"
    sae_cfg.metadata.hook_name = f"{cfg.representation}_layer_{cfg.layer:02d}"
    sae_cfg.metadata.dataset_path = str(cfg.activation_root)

    logging_cfg = LoggingConfig(
        log_to_wandb=cfg.log_to_wandb,
        log_weights_to_wandb=cfg.log_weights_to_wandb,
        log_optimizer_state_to_wandb=cfg.log_optimizer_state_to_wandb,
        wandb_project=cfg.wandb_project,
        wandb_id=cfg.wandb_id,
        run_name=run_name,
        wandb_entity=cfg.wandb_entity,
        wandb_log_frequency=cfg.wandb_log_frequency,
        eval_every_n_wandb_logs=cfg.eval_every_n_wandb_logs,
    )
    trainer_cfg = SAETrainerConfig(
        n_checkpoints=cfg.n_checkpoints,
        checkpoint_path=str(cfg.checkpoint_path or (run_root / "checkpoints")),
        save_final_checkpoint=cfg.save_final_checkpoint,
        total_training_samples=total_training_samples,
        device=cfg.device,
        autocast=cfg.autocast,
        lr=cfg.lr,
        lr_end=cfg.resolved_lr_end(),
        lr_scheduler_name=cfg.lr_scheduler_name,
        lr_warm_up_steps=cfg.lr_warm_up_steps,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        lr_decay_steps=cfg.lr_decay_steps or (total_training_steps // 5),
        n_restart_cycles=cfg.n_restart_cycles,
        train_batch_size_samples=cfg.train_batch_size_samples,
        dead_feature_window=cfg.dead_feature_window,
        feature_sampling_window=cfg.feature_sampling_window,
        n_batches_for_norm_estimate=cfg.n_batches_for_norm_estimate,
        logger=logging_cfg,
    )

    train_provider = NpzActivationDataProvider(
        train_chunks,
        batch_size=cfg.train_batch_size_samples,
        seed=cfg.seed,
        shuffle_chunks=True,
        shuffle_rows=True,
    )
    evaluator = ChunkedActivationEvaluator(
        val_chunks,
        batch_size=cfg.train_batch_size_samples,
        max_batches=cfg.holdout_eval_batches,
        seed=cfg.seed + 1,
    )
    sae = TrainingSAE.from_dict(sae_cfg.to_dict()).to(cfg.device)
    trainer = SAETrainer(
        cfg=trainer_cfg,
        sae=sae,
        data_provider=train_provider,
        evaluator=evaluator if val_chunks else None,
    )

    wandb_run = None
    if cfg.log_to_wandb:
        wandb_run = start_wandb_run(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            run_name=run_name,
            run_id=cfg.wandb_id,
            group=wandb_group,
            config={
                "train_config": cfg.to_dict(),
                "d_in": layer_meta.d_in,
                "d_sae": d_sae,
                "train_chunks": len(train_chunks),
                "val_chunks": len(val_chunks),
            },
            tags=["gem", "sae", "jumprelu", f"layer:{cfg.layer:02d}", cfg.representation],
        )

    try:
        trained_sae = trainer.fit()
        final_dir = run_root / "final_sae"
        weights_path, cfg_path = trained_sae.save_inference_model(final_dir)

        summary = {
            "run_name": run_name,
            "representation": cfg.representation,
            "layer": cfg.layer,
            "d_in": layer_meta.d_in,
            "d_sae": d_sae,
            "available_training_samples": available_training_samples,
            "used_training_samples": total_training_samples,
            "train_chunks": len(train_chunks),
            "val_chunks": len(val_chunks),
            "final_sae_dir": str(final_dir),
            "wandb_run_url": wandb_run.url if wandb_run else None,
        }
        save_json(run_root / "train_config.json", cfg.to_dict())
        save_json(run_root / "train_summary.json", summary)

        if wandb_run:
            save_wandb_metadata(run_root / "wandb_run.json", wandb_run)
            for key, value in summary.items():
                wandb_run.summary[key] = value
            if trainer.feature_sparsity.numel() > 0:
                wandb_run.summary["final_mean_log10_feature_sparsity"] = float(
                    trainer.log_feature_sparsity.mean().item()
                )
                wandb_run.summary["final_dead_features"] = int(
                    trainer.dead_neurons.sum().item()
                )
            log_wandb_artifact(
                name=run_name.replace("/", "_"),
                artifact_type="sae_model",
                files=[Path(weights_path), Path(cfg_path), run_root / "train_config.json"],
                metadata=summary,
                aliases=["latest", "final"],
            )

        if wandb_run:
            print(f"wandb_run_url={wandb_run.url}")
        return summary
    finally:
        if cfg.log_to_wandb:
            finish_wandb_run()
