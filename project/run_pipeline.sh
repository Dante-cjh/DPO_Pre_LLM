#!/usr/bin/env bash
# run_pipeline.sh — Revised v0.2 pipeline (non-heuristic prompt policy)
#
# Run from the project/ directory:
#   cd <LLaMA-Factory-root>/project
#   bash run_pipeline.sh
#
# Prerequisites:
#   - PHEME data processed to data/processed/{train,val,test}.jsonl
#   - Qwen3-1.7B and Qwen3-8B available locally or via HuggingFace
#   - conda/venv with: transformers peft torch scikit-learn openai pyyaml python-dotenv
#
# Experiment groups produced:
#   G0  BasePack-Only (also serves as frozen SLM probe for reward)
#   G1  Heuristic Pre (reproduced on BasePack-v2)
#   G2  SFT Hint Pre
#   G3  DPO Hint Pre  [main result]
#   G3-abl-R  DPO with LLM-only reward   (ablation)
#   G3-abl-C  DPO with template-only candidates (ablation)

set -euo pipefail
LLAMAFACTORY_ROOT="$(dirname "$(pwd)")"

# SLM 训练默认跳过已有完整产物的目录；强制全部重训: FORCE_TRAIN_SLM=1 bash run_pipeline.sh
FORCE_TRAIN_SLM="${FORCE_TRAIN_SLM:-0}"
if [ "${FORCE_TRAIN_SLM}" = "1" ]; then
    _train_slm_extra="--force"
else
    _train_slm_extra=""
fi

# 若候选文件已完整生成（行数与 basepack 对齐），则跳过 Step 7；强制重跑: FORCE_CANDIDATES=1
FORCE_CANDIDATES="${FORCE_CANDIDATES:-0}"
FORCE_SFT_DATA="${FORCE_SFT_DATA:-0}"

# GPU 策略：推理/数据构建默认单卡，SFT/DPO 训练使用双卡
CUDA_SINGLE="${CUDA_SINGLE:-5}"
CUDA_MULTI="${CUDA_MULTI:-4,5}"
export CUDA_VISIBLE_DEVICES="${CUDA_SINGLE}"

# RTX 40 系列 + 旧驱动环境下，禁用 NCCL P2P/IB 避免通信问题
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

is_complete_jsonl_for_split() {
    local split="$1"
    local basepack_path="${BASEPACK_V2_DIR}/basepack_${split}.jsonl"
    local out_path="policy_data/dpo/candidate_hints_${split}.jsonl"

    if [ ! -f "${basepack_path}" ] || [ ! -f "${out_path}" ]; then
        return 1
    fi

    local base_n out_n
    base_n=$(wc -l < "${basepack_path}")
    out_n=$(wc -l < "${out_path}")
    [ "${base_n}" -gt 0 ] && [ "${out_n}" -ge "${base_n}" ]
}

is_sft_split_complete() {
    local split="$1"
    local bp_path="${BASEPACK_V2_DIR}/basepack_${split}.jsonl"
    local sft_path="policy_data/sft/${split}.json"

    if [ ! -f "${bp_path}" ] || [ ! -f "${sft_path}" ]; then
        return 1
    fi

    python - "${bp_path}" "${sft_path}" <<'PY'
import json
from pathlib import Path
import sys

bp_path = Path(sys.argv[1])
sft_path = Path(sys.argv[2])

def jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def json_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data) if isinstance(data, list) else -1

try:
    ok = jsonl_count(bp_path) > 0 and json_count(sft_path) == jsonl_count(bp_path)
except Exception:
    ok = False

sys.exit(0 if ok else 1)
PY
}

# ── configurable paths ─────────────────────────────────────────────────────────
QWEN_1P7B="${QWEN_1P7B:-Qwen/Qwen3-1.7B}"
QWEN_8B="${QWEN_8B:-Qwen/Qwen3-8B}"
BASEPACK_V2_DIR="basepack_v2"
# ──────────────────────────────────────────────────────────────────────────────

# ── vLLM 配置 ──────────────────────────────────────────────────────────────────
# 用哪张卡跑 vLLM（单卡即可，4090 无 NVLink 不建议 tensor-parallel）
VLLM_GPU="${VLLM_GPU:-5}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
VLLM_MODEL_NAME="Qwen3-8B"          # vLLM served-model-name，供客户端调用
VLLM_STARTUP_TIMEOUT=180            # 等待 vLLM 就绪的最长秒数
VLLM_API_WORKERS="${VLLM_API_WORKERS:-50}"  # 并发请求数
VLLM_PID=""                         # 由本脚本启动的 vLLM 进程 PID
VLLM_STARTED_HERE=false             # 标记是否是本脚本启动的

# 确保 vLLM 服务在运行；若未运行则自动启动
ensure_vllm_running() {
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "  [vLLM] 服务已在 port=${VLLM_PORT} 运行，直接复用。"
        return 0
    fi

    echo "  [vLLM] 服务未运行，正在 GPU=${VLLM_GPU} 上启动..."
    mkdir -p logs
    CUDA_VISIBLE_DEVICES="${VLLM_GPU}" nohup vllm serve "${QWEN_8B}" \
        --port "${VLLM_PORT}" \
        --max-model-len 2048 \
        --served-model-name "${VLLM_MODEL_NAME}" \
        > logs/vllm.log 2>&1 &
    VLLM_PID=$!
    VLLM_STARTED_HERE=true
    echo "  [vLLM] 进程已启动 (PID=${VLLM_PID})，等待 health check..."

    local elapsed=0
    until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ "${elapsed}" -ge "${VLLM_STARTUP_TIMEOUT}" ]; then
            echo "[ERROR] vLLM 在 ${VLLM_STARTUP_TIMEOUT}s 内未能就绪，请检查 logs/vllm.log"
            exit 1
        fi
    done
    echo "  [vLLM] 服务就绪（等待了 ${elapsed}s）。"
}

# 若 vLLM 是本脚本启动的，则在不再需要时停止它以释放显存
stop_vllm_if_we_started() {
    if [ "${VLLM_STARTED_HERE}" = "true" ] && [ -n "${VLLM_PID}" ]; then
        echo "  [vLLM] 停止本脚本启动的 vLLM 进程 (PID=${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null && wait "${VLLM_PID}" 2>/dev/null || true
        VLLM_PID=""
        VLLM_STARTED_HERE=false
        echo "  [vLLM] 已停止，GPU 显存已释放。"
    fi
}
# ──────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo " Step 0: Build BasePack-v1 (for G0/G1 reproducibility baseline)"
echo "================================================================"
python scripts/01_build_basepack.py \
    --input_dir data/processed \
    --output_dir basepack \
    --max_replies 8

echo ""
echo "================================================================"
echo " Step 1: Build BasePack-v2-Stance (main view for G2/G3)"
echo "================================================================"
python scripts/01b_build_basepack_v2.py \
    --input_dir data/processed \
    --output_dir "${BASEPACK_V2_DIR}" \
    --stance_model "${QWEN_1P7B}"

echo ""
echo "================================================================"
echo " Step 2: Build G0 SLM dataset (BasePack-v2, no LLM_AUG)"
echo "================================================================"
# G0 uses v2 basepack directly as SLM input
for split in train val test; do
    python scripts/08_build_slm_dataset.py \
        --basepack "${BASEPACK_V2_DIR}/basepack_${split}.jsonl" \
        --output "slm_data/basepack_only/${split}.jsonl"
done

echo ""
echo "================================================================"
echo " Step 3: Train G0 SLM (DeBERTa-v3-base, BasePack-only)"
echo "         G0 checkpoint also serves as the frozen SLM probe"
echo "================================================================"
python scripts/09_train_slm.py --method basepack_only ${_train_slm_extra}

G0_PROBE="slm_outputs/basepack_only/best_model"
echo "  G0 SLM probe ready at: ${G0_PROBE}"

echo ""
echo "================================================================"
echo " Step 4: Build SFT policy training data"
echo "         ~70% LLM-rewritten (Qwen3-8B) + 30% template"
echo "================================================================"
pending_sft_splits=()
for split in train val; do
    if [ "${FORCE_SFT_DATA}" != "1" ] && is_sft_split_complete "${split}"; then
        echo "  Skip Step 4 (${split}): policy_data/sft/${split}.json already complete."
    else
        pending_sft_splits+=("${split}")
    fi
done

if [ ${#pending_sft_splits[@]} -eq 0 ]; then
    echo "  Skip Step 4: all requested splits are already complete."
else
    # 单行调用，避免续行符丢失导致「--template_ratio: command not found」
    CUDA_VISIBLE_DEVICES="${CUDA_MULTI}" python scripts/02_make_sft_policy_data.py --basepack_dir "${BASEPACK_V2_DIR}" --output_dir policy_data/sft --llm_model "${QWEN_8B}" --template_ratio 0.3 --splits "${pending_sft_splits[@]}"
fi

echo ""
echo " Step 4b: Copy SFT data to LLaMA-Factory data dir"
cp policy_data/sft/train.json "${LLAMAFACTORY_ROOT}/data/pheme_hint_sft.json"
echo "  Copied -> ${LLAMAFACTORY_ROOT}/data/pheme_hint_sft.json"

echo ""
echo "================================================================"
echo " Step 5: Train SFT policy (Qwen3-1.7B + LoRA)"
echo "================================================================"
cd "${LLAMAFACTORY_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_MULTI}" llamafactory-cli train project/llamafactory_configs/qwen3_1p7b_lora_sft.yaml
cd project

echo ""
echo "================================================================"
echo " Step 6: Generate SFT policy hints for all splits (G2 inference)"
echo "================================================================"
python scripts/06_generate_policy_hints.py \
    --policy sft \
    --base_model "${QWEN_1P7B}"

echo ""
echo "================================================================"
echo " Step 7: Generate 12 candidate hints per event"
echo "         4 template + 6 LLM-rewritten (Qwen3-8B) + 2 SFT samples"
echo "================================================================"
for split in train val; do
    if [ "${FORCE_CANDIDATES}" != "1" ] && is_complete_jsonl_for_split "${split}"; then
        echo "  Skip Step 7 (${split}): policy_data/dpo/candidate_hints_${split}.jsonl already complete."
    else
        python scripts/03_make_candidate_hints.py \
            --split "${split}" \
            --llm_model "${QWEN_8B}" \
            --sft_base_model "${QWEN_1P7B}"
    fi
done

echo ""
echo "================================================================"
echo " Step 8: Score candidates with SLM-aligned reward"
echo "         Uses vLLM Qwen3-8B (API) + frozen G0 SLM probe"
echo "================================================================"
ensure_vllm_running

export CUDA_VISIBLE_DEVICES="${CUDA_SINGLE}"
python scripts/04_score_candidate_hints.py \
    --split train \
    --backend api \
    --api_base_url "${VLLM_BASE_URL}" \
    --api_model "${VLLM_MODEL_NAME}" \
    --api_key EMPTY \
    --api_workers "${VLLM_API_WORKERS}" \
    --slm_probe "${G0_PROBE}"

python scripts/04_score_candidate_hints.py \
    --split val \
    --backend api \
    --api_base_url "${VLLM_BASE_URL}" \
    --api_model "${VLLM_MODEL_NAME}" \
    --api_key EMPTY \
    --api_workers "${VLLM_API_WORKERS}" \
    --slm_probe "${G0_PROBE}"

echo ""
echo "================================================================"
echo " Step 9: Build DPO pairs (min_margin=1.0, within-event norm)"
echo "================================================================"
python scripts/05_make_dpo_pairs.py \
    --scored policy_data/dpo/scored_candidates_train.jsonl \
    --basepack "${BASEPACK_V2_DIR}/basepack_train.jsonl" \
    --output policy_data/dpo/train.json \
    --min_margin 1.0

echo ""
echo " Step 9b: Copy DPO data to LLaMA-Factory data dir"
cp policy_data/dpo/train.json "${LLAMAFACTORY_ROOT}/data/pheme_hint_dpo.json"
echo "  Copied -> ${LLAMAFACTORY_ROOT}/data/pheme_hint_dpo.json"

echo ""
echo "================================================================"
echo " Step 10: Train DPO policy (Qwen3-1.7B + LoRA, SLM-aligned)"
echo "================================================================"
cd "${LLAMAFACTORY_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_MULTI}" llamafactory-cli train project/llamafactory_configs/qwen3_1p7b_lora_dpo.yaml
cd project

echo ""
echo "================================================================"
echo " Step 11: Generate DPO policy hints for all splits (G3 inference)"
echo "================================================================"
python scripts/06_generate_policy_hints.py \
    --policy dpo \
    --base_model "${QWEN_1P7B}"

echo ""
echo "================================================================"
echo " Step 12: Generate LLM_AUG for G1/G2/G3"
echo "          All methods use vLLM Qwen3-8B (API, concurrent)"
echo "          vLLM is (re-)started here if it was stopped after Step 8"
echo "================================================================"
ensure_vllm_running

export CUDA_VISIBLE_DEVICES="${CUDA_SINGLE}"
python scripts/07_run_qwen_turbo_aug.py \
    --method heuristic \
    --backend api \
    --api_base_url "${VLLM_BASE_URL}" \
    --api_model "${VLLM_MODEL_NAME}" \
    --api_key EMPTY \
    --api_workers "${VLLM_API_WORKERS}" \
    --basepack_dir "${BASEPACK_V2_DIR}"

python scripts/07_run_qwen_turbo_aug.py \
    --method sft_hint \
    --backend api \
    --api_base_url "${VLLM_BASE_URL}" \
    --api_model "${VLLM_MODEL_NAME}" \
    --api_key EMPTY \
    --api_workers "${VLLM_API_WORKERS}" \
    --basepack_dir "${BASEPACK_V2_DIR}"

python scripts/07_run_qwen_turbo_aug.py \
    --method dpo_hint \
    --backend api \
    --api_base_url "${VLLM_BASE_URL}" \
    --api_model "${VLLM_MODEL_NAME}" \
    --api_key EMPTY \
    --api_workers "${VLLM_API_WORKERS}" \
    --basepack_dir "${BASEPACK_V2_DIR}"

stop_vllm_if_we_started

echo ""
echo "================================================================"
echo " Step 13: Build SLM datasets for G1/G2/G3"
echo "================================================================"
python scripts/08_build_slm_dataset.py --method heuristic_pre
python scripts/08_build_slm_dataset.py --method sft_hint_pre
python scripts/08_build_slm_dataset.py --method dpo_hint_pre

echo ""
echo "================================================================"
echo " Step 14: Train SLM classifiers for G1/G2/G3"
echo "================================================================"
python scripts/09_train_slm.py --method heuristic_pre ${_train_slm_extra}
python scripts/09_train_slm.py --method sft_hint_pre ${_train_slm_extra}
python scripts/09_train_slm.py --method dpo_hint_pre ${_train_slm_extra}

echo ""
echo "================================================================"
echo " Step 15: Evaluate all groups and report"
echo "================================================================"
python scripts/10_eval.py

echo ""
echo "================================================================"
echo " ABLATION: G3-abl-C — DPO with template-only candidates"
echo "           Re-run steps 7-14 with --no_llm flag"
echo "================================================================"
echo "  To run G3-abl-C ablation:"
echo "    python scripts/03_make_candidate_hints.py --split train --no_llm --no_sft"
echo "    python scripts/04_score_candidate_hints.py --split train ..."
echo "    python scripts/05_make_dpo_pairs.py --output policy_data/dpo_abl_c/train.json"
echo "    (then retrain DPO policy and evaluate as dpo_hint_abl_c)"

echo ""
echo "================================================================"
echo " Pipeline complete. Results in outputs/metrics/"
echo "================================================================"
