#!/usr/bin/env bash
# 0a_build_reward_probe.sh
# Build an aug-aware SLM probe before DPO scoring.
#
# Usage, from project/:
#   bash scripts/0a_build_reward_probe.sh

set -euo pipefail

QWEN_8B="${QWEN_8B:-Qwen/Qwen3-8B}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen3-8B}"
VLLM_API_WORKERS="${VLLM_API_WORKERS:-50}"
CUDA_SINGLE="${CUDA_SINGLE:-5}"

echo "================================================================"
echo " [reward-probe] Generate heuristic LLM_AUG for train/val/test"
echo "================================================================"
export CUDA_VISIBLE_DEVICES="${CUDA_SINGLE}"
python scripts/07_run_qwen_turbo_aug.py \
    --method heuristic \
    --backend api \
    --api_base_url "${VLLM_BASE_URL}" \
    --api_model "${VLLM_MODEL_NAME}" \
    --api_key EMPTY \
    --api_workers "${VLLM_API_WORKERS}" \
    --basepack_dir basepack_v2 \
    --splits train val test

echo ""
echo "================================================================"
echo " [reward-probe] Build SLM dataset (basepack_v2 + heuristic LLM_AUG)"
echo "================================================================"
python scripts/08_build_slm_dataset.py --method heuristic_pre

echo ""
echo "================================================================"
echo " [reward-probe] Train SLM probe (DeBERTa-v3-base, aug-aware)"
echo "================================================================"
python scripts/09_train_slm.py \
    --train slm_data/heuristic_pre/train.jsonl \
    --val   slm_data/heuristic_pre/val.jsonl \
    --test  slm_data/heuristic_pre/test.jsonl \
    --output_dir slm_outputs/reward_probe \
    --force

echo ""
echo "[reward-probe] Done. Use --slm_probe slm_outputs/reward_probe/best_model in step 8."
