#!/usr/bin/env bash
# 논문 SAE 학습 (vocfix-graph-L08-x32-l0_0.001-30ep 재현).
# 실행 위치: PaddleHelix/apps/pretrained_compound/ChemRL/GEM (이 디렉토리의 파일들을 그 안에 둔 상태)
# 사용: ./run_paper_sae.sh [GPU_ID]   환경변수 ACTROOT / OUTDIR 로 경로 지정
set -euo pipefail

GPU_ID="${1:-0}"
shift || true
ACTROOT="${ACTROOT:-artifacts/gem_extract_zinc15_2p2m}"   # extract_activations.py 출력
OUTDIR="${OUTDIR:-sae_models}"

# 추출이 끝났는지 확인 (validation split에 chunk 2개 이상 필요)
NCHUNKS=$(ls "$ACTROOT/graph/layer_08/chunk_"*.npz 2>/dev/null | wc -l)
if [ "$NCHUNKS" -lt 2 ]; then
    echo "ERR: need >=2 graph chunks under $ACTROOT/graph/layer_08 (got $NCHUNKS)" >&2
    exit 1
fi

# 66,000,000 samples = 2.2M 분자 × 30 epoch
CUDA_VISIBLE_DEVICES="$GPU_ID" python train_sae.py \
    --activation-root "$ACTROOT" \
    --output-dir "$OUTDIR" \
    --layer 8 \
    --representation graph \
    --d-sae 1024 \
    --total-training-samples 66000000 \
    --train-batch-size-samples 4096 \
    --validation-fraction 0.02 \
    --holdout-eval-batches 8 \
    --lr 3e-4 \
    --lr-scheduler-name constant \
    --normalize-activations expected_average_only_in \
    --jumprelu-sparsity-loss-mode tanh \
    --jumprelu-tanh-scale 4.0 \
    --jumprelu-bandwidth 0.05 \
    --jumprelu-init-threshold 0.1 \
    --jumprelu-l0-coefficient 0.001 \
    --jumprelu-pre-act-loss-coefficient 3e-6 \
    --seed 42 \
    --run-name vocfix-graph-L08-x32-l0_0.001-30ep \
    "$@"
