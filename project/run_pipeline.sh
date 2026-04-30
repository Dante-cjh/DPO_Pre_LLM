#!/usr/bin/env bash
# run_pipeline.sh
# Full MVP experiment pipeline for Prompt Policy training.
#
# Run from the project/ directory:
#   cd <LLaMA-Factory-root>/project
#   bash run_pipeline.sh
#
# Prerequisites:
#   - export DASHSCOPE_API_KEY="sk-..."
#   - PHEME data already processed to data/processed/{train,val,test}.jsonl
#     (run FN_Cascaded scripts 01_build_pheme_binary.py + 02_make_splits.py first)
#   - Qwen3-1.7B available at local path or reachable via HuggingFace
#   - conda/venv with: transformers peft torch openai pyyaml scikit-learn
#                      deberta (pip install deberta) or transformers[sentencepiece]

set -euo pipefail
LLAMAFACTORY_ROOT="$(dirname "$(pwd)")"   # one level up from project/

echo "===  Step 1: Build BasePack  ==="
python scripts/01_build_basepack.py

echo "===  Step 2: Build SFT policy data  ==="
python scripts/02_make_sft_policy_data.py

echo "===  Step 3: Copy SFT data to LLaMA-Factory data dir  ==="
cp policy_data/sft/train.json "${LLAMAFACTORY_ROOT}/data/pheme_hint_sft.json"
echo "  Copied -> ${LLAMAFACTORY_ROOT}/data/pheme_hint_sft.json"

echo "===  Step 4: Train SFT policy (Qwen3-1.7B + LoRA)  ==="
cd "${LLAMAFACTORY_ROOT}"
llamafactory-cli train project/llamafactory_configs/qwen3_1p7b_lora_sft.yaml
cd project

echo "===  Step 5: Generate candidate hints  ==="
python scripts/03_make_candidate_hints.py --split train
python scripts/03_make_candidate_hints.py \
    --split val \
    --output policy_data/dpo/candidate_hints_val.jsonl

echo "===  Step 6: Score candidate hints via Qwen3-turbo  ==="
python scripts/04_score_candidate_hints.py --split train
python scripts/04_score_candidate_hints.py --split val

echo "===  Step 7: Build DPO pairs  ==="
python scripts/05_make_dpo_pairs.py

echo "===  Step 8: Copy DPO data to LLaMA-Factory data dir  ==="
cp policy_data/dpo/train.json "${LLAMAFACTORY_ROOT}/data/pheme_hint_dpo.json"
echo "  Copied -> ${LLAMAFACTORY_ROOT}/data/pheme_hint_dpo.json"

echo "===  Step 9: Train DPO policy  ==="
cd "${LLAMAFACTORY_ROOT}"
llamafactory-cli train project/llamafactory_configs/qwen3_1p7b_lora_dpo.yaml
cd project

echo "===  Step 10: Generate SFT and DPO policy hints for all splits  ==="
python scripts/06_generate_policy_hints.py --policy sft
python scripts/06_generate_policy_hints.py --policy dpo

echo "===  Step 11: Generate LLM_AUG for all three hint methods  ==="
python scripts/07_run_qwen_turbo_aug.py --method heuristic
python scripts/07_run_qwen_turbo_aug.py --method sft_hint
python scripts/07_run_qwen_turbo_aug.py --method dpo_hint

echo "===  Step 12: Build SLM datasets  ==="
python scripts/08_build_slm_dataset.py --method basepack_only
python scripts/08_build_slm_dataset.py --method heuristic_pre
python scripts/08_build_slm_dataset.py --method sft_hint_pre
python scripts/08_build_slm_dataset.py --method dpo_hint_pre

echo "===  Step 13: Train four SLM classifiers  ==="
python scripts/09_train_slm.py --method basepack_only
python scripts/09_train_slm.py --method heuristic_pre
python scripts/09_train_slm.py --method sft_hint_pre
python scripts/09_train_slm.py --method dpo_hint_pre

echo "===  Step 14: Aggregate and report results  ==="
python scripts/10_eval.py

echo ""
echo "=== Pipeline complete. Results in outputs/metrics/ ==="
