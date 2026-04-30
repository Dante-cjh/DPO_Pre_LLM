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

# ── configurable paths ─────────────────────────────────────────────────────────
QWEN_1P7B="${QWEN_1P7B:-Qwen/Qwen3-1.7B}"
QWEN_8B="${QWEN_8B:-Qwen/Qwen3-8B}"
BASEPACK_V2_DIR="basepack_v2"
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
python scripts/09_train_slm.py --method basepack_only

G0_PROBE="slm_outputs/basepack_only/best_model"
echo "  G0 SLM probe ready at: ${G0_PROBE}"

echo ""
echo "================================================================"
echo " Step 4: Build SFT policy training data"
echo "         ~70% LLM-rewritten (Qwen3-8B) + 30% template"
echo "================================================================"
python scripts/02_make_sft_policy_data.py \
    --basepack_dir "${BASEPACK_V2_DIR}" \
    --output_dir policy_data/sft \
    --llm_model "${QWEN_8B}" \
    --template_ratio 0.3

echo ""
echo " Step 4b: Copy SFT data to LLaMA-Factory data dir"
cp policy_data/sft/train.json "${LLAMAFACTORY_ROOT}/data/pheme_hint_sft.json"
echo "  Copied -> ${LLAMAFACTORY_ROOT}/data/pheme_hint_sft.json"

echo ""
echo "================================================================"
echo " Step 5: Train SFT policy (Qwen3-1.7B + LoRA)"
echo "================================================================"
cd "${LLAMAFACTORY_ROOT}"
llamafactory-cli train project/llamafactory_configs/qwen3_1p7b_lora_sft.yaml
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
python scripts/03_make_candidate_hints.py \
    --split train \
    --llm_model "${QWEN_8B}" \
    --sft_base_model "${QWEN_1P7B}"

python scripts/03_make_candidate_hints.py \
    --split val \
    --llm_model "${QWEN_8B}" \
    --sft_base_model "${QWEN_1P7B}"

echo ""
echo "================================================================"
echo " Step 8: Score candidates with SLM-aligned reward"
echo "         Uses local Qwen3-8B + frozen G0 SLM probe"
echo "================================================================"
python scripts/04_score_candidate_hints.py \
    --split train \
    --llm_model "${QWEN_8B}" \
    --slm_probe "${G0_PROBE}"

python scripts/04_score_candidate_hints.py \
    --split val \
    --llm_model "${QWEN_8B}" \
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
llamafactory-cli train project/llamafactory_configs/qwen3_1p7b_lora_dpo.yaml
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
echo "          All methods use local Qwen3-8B"
echo "================================================================"
python scripts/07_run_qwen_turbo_aug.py \
    --method heuristic \
    --backend local \
    --llm_model "${QWEN_8B}" \
    --basepack_dir "${BASEPACK_V2_DIR}"

python scripts/07_run_qwen_turbo_aug.py \
    --method sft_hint \
    --backend local \
    --llm_model "${QWEN_8B}" \
    --basepack_dir "${BASEPACK_V2_DIR}"

python scripts/07_run_qwen_turbo_aug.py \
    --method dpo_hint \
    --backend local \
    --llm_model "${QWEN_8B}" \
    --basepack_dir "${BASEPACK_V2_DIR}"

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
python scripts/09_train_slm.py --method heuristic_pre
python scripts/09_train_slm.py --method sft_hint_pre
python scripts/09_train_slm.py --method dpo_hint_pre

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
