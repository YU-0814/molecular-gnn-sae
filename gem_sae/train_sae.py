#!/usr/bin/env python3
"""Train a SAELens JumpReLU SAE on extracted GEM activations. Defaults = paper setting (graph, 32->1024, l0 coef 0.001, batch 4096, lr 3e-4); see run_paper_sae.sh."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a SAELens JumpReLU SAE on cached GEM activations.",
    )
    parser.add_argument("--activation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--representation", choices=("node", "graph"), default="graph")
    parser.add_argument("--saelens-root", type=Path, default=None)
    parser.add_argument("--d-sae", type=int, default=None)
    parser.add_argument("--expansion-factor", type=int, default=32)  # 32 → 1024
    parser.add_argument("--total-training-samples", type=int, default=None)
    parser.add_argument("--train-batch-size-samples", type=int, default=4096)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--holdout-eval-batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-end", type=float, default=None)
    parser.add_argument(
        "--lr-scheduler-name",
        choices=("constant", "cosineannealing", "cosineannealingwarmrestarts"),
        default="constant",
    )
    parser.add_argument("--lr-warm-up-steps", type=int, default=0)
    parser.add_argument("--lr-decay-steps", type=int, default=0)
    parser.add_argument("--n-restart-cycles", type=int, default=1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--dead-feature-window", type=int, default=1000)
    parser.add_argument("--feature-sampling-window", type=int, default=2000)
    parser.add_argument(
        "--n-batches-for-norm-estimate",
        type=int,
        default=1000,
        help="batches consumed for SAELens norm scaling factor estimate at start of training. "
             "Lower this for faster startup with large train_batch_size_samples.",
    )
    parser.add_argument("--n-checkpoints", type=int, default=1)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--disable-final-checkpoint", action="store_true")
    parser.add_argument(
        "--normalize-activations",
        choices=("none", "expected_average_only_in", "layer_norm"),
        default="expected_average_only_in",
    )
    parser.add_argument(
        "--jumprelu-sparsity-loss-mode",
        choices=("step", "tanh"),
        default="tanh",
    )
    parser.add_argument("--jumprelu-tanh-scale", type=float, default=4.0)
    parser.add_argument("--jumprelu-bandwidth", type=float, default=0.05)
    parser.add_argument("--jumprelu-init-threshold", type=float, default=0.1)
    parser.add_argument("--jumprelu-l0-coefficient", type=float, default=0.001)
    parser.add_argument("--jumprelu-l0-warm-up-steps", type=int, default=0)
    parser.add_argument("--jumprelu-pre-act-loss-coefficient", type=float, default=3e-6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-autocast", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gem_sae")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb-log-frequency", type=int, default=10)
    parser.add_argument("--eval-every-n-wandb-logs", type=int, default=20)
    parser.add_argument("--disable-log-weights-to-wandb", action="store_true")
    parser.add_argument("--log-optimizer-state-to-wandb", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from src.sae_config import GemSaeTrainConfig
    from src.sae_training import run_gem_sae_training

    cfg = GemSaeTrainConfig(
        activation_root=args.activation_root,
        output_dir=args.output_dir,
        layer=args.layer,
        representation=args.representation,
        saelens_root=args.saelens_root,
        d_sae=args.d_sae,
        expansion_factor=args.expansion_factor,
        total_training_samples=args.total_training_samples,
        train_batch_size_samples=args.train_batch_size_samples,
        validation_fraction=args.validation_fraction,
        holdout_eval_batches=args.holdout_eval_batches,
        lr=args.lr,
        lr_end=args.lr_end,
        lr_scheduler_name=args.lr_scheduler_name,
        lr_warm_up_steps=args.lr_warm_up_steps,
        lr_decay_steps=args.lr_decay_steps,
        n_restart_cycles=args.n_restart_cycles,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        dead_feature_window=args.dead_feature_window,
        feature_sampling_window=args.feature_sampling_window,
        n_batches_for_norm_estimate=args.n_batches_for_norm_estimate,
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=args.checkpoint_path,
        save_final_checkpoint=not args.disable_final_checkpoint,
        normalize_activations=args.normalize_activations,
        jumprelu_sparsity_loss_mode=args.jumprelu_sparsity_loss_mode,
        jumprelu_tanh_scale=args.jumprelu_tanh_scale,
        jumprelu_bandwidth=args.jumprelu_bandwidth,
        jumprelu_init_threshold=args.jumprelu_init_threshold,
        jumprelu_l0_coefficient=args.jumprelu_l0_coefficient,
        jumprelu_l0_warm_up_steps=args.jumprelu_l0_warm_up_steps,
        jumprelu_pre_act_loss_coefficient=args.jumprelu_pre_act_loss_coefficient,
        device=args.device,
        autocast=not args.disable_autocast,
        seed=args.seed,
        log_to_wandb=not args.disable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_id=args.wandb_id,
        wandb_group=args.wandb_group,
        run_name=args.run_name,
        wandb_log_frequency=args.wandb_log_frequency,
        eval_every_n_wandb_logs=args.eval_every_n_wandb_logs,
        log_weights_to_wandb=not args.disable_log_weights_to_wandb,
        log_optimizer_state_to_wandb=args.log_optimizer_state_to_wandb,
    )
    summary = run_gem_sae_training(cfg)
    print(summary)


if __name__ == "__main__":
    main()
