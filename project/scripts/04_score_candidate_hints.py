"""
04_score_candidate_hints.py
Score each candidate hint using SLM-aligned reward.

Pipeline per (event, hint):
  1. Call local Qwen3-8B with the hint to produce LLM_AUG (JSON analysis)
  2. Forward BasePack through frozen G0 SLM probe -> p_base[gold]
  3. Forward BasePack + LLM_AUG through frozen G0 SLM probe -> p_aug[gold]
  4. Compute slm_gain = log(p_aug + eps) - log(p_base + eps)
  5. Compute full reward:
       1.5 * slm_gain
     + 0.5 * acc_llm            (weak_label == gold)
     + 0.3 * format_ok          (valid JSON with all 6 fields)
     + 0.2 * field_coverage      (fraction of 6 fields non-empty)
     - 0.1 * log(1 + out_tok/100)

Input:  policy_data/dpo/candidate_hints_{split}.jsonl
        slm_outputs/basepack_only/best_model/  (frozen G0 SLM probe)
Output: policy_data/dpo/scored_candidates_{split}.jsonl
"""

import json
import math
import re
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer as HFAutoTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel

REQUIRED_FIELDS = [
    "claim_summary", "supporting_signals", "refuting_signals",
    "conflict_summary", "verification_focus", "uncertainty_note",
]

ADVISOR_SYSTEM_PROMPT = (
    "You are a misinformation analysis assistant. Analyze the given social media event and "
    "follow the provided focus hint. Return a JSON object with exactly these fields:\n"
    "claim_summary, supporting_signals, refuting_signals, conflict_summary, "
    "verification_focus, uncertainty_note, weak_label (\"true\" or \"fake\"), weak_confidence (0.5-1.0).\n"
    "Only use information present in the event. Do not invent facts."
)

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
EPS = 1e-6
SLM_MAX_LENGTH = 512


def load_llm_8b(model_name: str):
    print(f"Loading Qwen3-8B: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def load_slm_probe(model_path: str):
    print(f"Loading frozen SLM probe: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        torch_dtype=torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model, device


def format_llm_aug(aug: dict) -> str:
    lines = ["[LLM_AUG]"]
    if aug.get("claim_summary"):
        lines.append(f"Claim: {aug['claim_summary']}")
    sup = aug.get("supporting_signals", [])
    if sup:
        lines.append(f"Supporting: {' | '.join(str(s) for s in sup)}")
    ref = aug.get("refuting_signals", [])
    if ref:
        lines.append(f"Refuting: {' | '.join(str(s) for s in ref)}")
    if aug.get("conflict_summary"):
        lines.append(f"Conflict: {aug['conflict_summary']}")
    vf = aug.get("verification_focus", [])
    if vf:
        lines.append(f"VerificationFocus: {' | '.join(str(v) for v in vf)}")
    if aug.get("uncertainty_note"):
        lines.append(f"Uncertainty: {aug['uncertainty_note']}")
    return "\n".join(lines)


def generate_llm_aug(llm_tokenizer, llm_model,
                     focus_hint: str, basepack_text: str,
                     max_new_tokens: int = 300) -> tuple[dict | None, str | None, int]:
    user_msg = (
        f"[FOCUS_HINT]\n{focus_hint}\n\n"
        f"[EVENT]\n{basepack_text}"
    )
    messages = [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    text = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=llm_tokenizer.pad_token_id,
        )
    out_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
    raw = llm_tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    parsed = _parse_aug_response(raw)
    return parsed, raw, out_tokens


def _parse_aug_response(raw: str) -> dict | None:
    clean = raw.strip()
    m = JSON_BLOCK_RE.search(clean)
    if m:
        clean = m.group(1)
    else:
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1:
            clean = clean[start:end + 1]
    try:
        obj = json.loads(clean)
    except json.JSONDecodeError:
        return None
    if not all(f in obj for f in REQUIRED_FIELDS):
        return None
    if str(obj.get("weak_label", "")).lower() not in ("true", "fake"):
        return None
    try:
        float(obj["weak_confidence"])
    except (ValueError, TypeError):
        return None
    return obj


def slm_probe_log_prob(slm_tokenizer, slm_model, device: str,
                       text: str, gold_label: int) -> float:
    enc = slm_tokenizer(
        text,
        truncation=True,
        max_length=SLM_MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = slm_model(**enc).logits  # (1, 2)
    log_probs = F.log_softmax(logits, dim=-1)
    return float(log_probs[0, gold_label].cpu())


def compute_slm_gain(slm_tokenizer, slm_model, device: str,
                     basepack_text: str, aug_text: str | None,
                     gold_label: int) -> float:
    base_input = f"[SOURCE_AND_REPLIES]\n{basepack_text}"
    log_p_base = slm_probe_log_prob(slm_tokenizer, slm_model, device, base_input, gold_label)

    if aug_text is None:
        return 0.0

    aug_input = f"[SOURCE_AND_REPLIES]\n{basepack_text}\n\n{aug_text}"
    log_p_aug = slm_probe_log_prob(slm_tokenizer, slm_model, device, aug_input, gold_label)

    return log_p_aug - log_p_base


def compute_reward(slm_gain: float, aug: dict | None, gold_label: int,
                   out_tokens: int) -> float:
    format_ok = 1.0 if aug is not None else 0.0

    field_coverage = 0.0
    acc_llm = 0.0
    if aug is not None:
        non_empty = sum(
            1 for f in REQUIRED_FIELDS
            if aug.get(f) and str(aug[f]).strip()
        )
        field_coverage = non_empty / len(REQUIRED_FIELDS)

        weak_label = str(aug.get("weak_label", "")).lower()
        label_map = {"true": 0, "fake": 1}
        pred = label_map.get(weak_label, -1)
        acc_llm = 1.0 if pred == gold_label else 0.0

    cost_penalty = math.log(1 + out_tokens / 100)

    return (
        1.5 * slm_gain
        + 0.5 * acc_llm
        + 0.3 * format_ok
        + 0.2 * field_coverage
        - 0.1 * cost_penalty
    )


def score_split(candidate_path: Path, output_path: Path,
                llm_model_name: str, slm_probe_path: str):
    llm_tokenizer, llm_model = load_llm_8b(llm_model_name)
    slm_tokenizer, slm_model, slm_device = load_slm_probe(slm_probe_path)

    events = []
    with open(candidate_path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    resume_keys: set = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    resume_keys.add((obj["event_id"], obj["hint_id"]))
        print(f"Resuming: {len(resume_keys)} records already scored.")

    total = sum(len(e["candidates"]) for e in events)
    done = 0

    with open(output_path, "a") as fout:
        for event in events:
            eid = event["event_id"]
            gold_label = int(event["label"])
            basepack_text = event["basepack_text"]

            for cand in event["candidates"]:
                hint_id = cand["hint_id"]
                key = (eid, hint_id)
                if key in resume_keys:
                    done += 1
                    continue

                focus_hint = cand["focus_hint"]

                aug, raw, out_tokens = generate_llm_aug(
                    llm_tokenizer, llm_model, focus_hint, basepack_text
                )

                aug_text = format_llm_aug(aug) if aug is not None else None
                slm_gain = compute_slm_gain(
                    slm_tokenizer, slm_model, slm_device,
                    basepack_text, aug_text, gold_label
                )

                reward = compute_reward(slm_gain, aug, gold_label, out_tokens)

                record = {
                    "event_id": eid,
                    "hint_id": hint_id,
                    "focus_hint": focus_hint,
                    "gold_label": gold_label,
                    "llm_json_valid": 1 if aug is not None else 0,
                    "weak_label": str(aug.get("weak_label", "")).lower() if aug else None,
                    "weak_confidence": float(aug.get("weak_confidence", 0.5)) if aug else None,
                    "slm_gain": round(slm_gain, 6),
                    "reward": round(reward, 6),
                    "output_tokens": out_tokens,
                    "analysis": {k: aug[k] for k in REQUIRED_FIELDS} if aug else None,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                done += 1

                if done % 50 == 0:
                    print(f"  Progress: {done}/{total}")

    print(f"\nDone. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Score candidate hints (SLM-aligned reward)")
    parser.add_argument("--candidates", default="policy_data/dpo/candidate_hints_train.jsonl")
    parser.add_argument("--output", default="policy_data/dpo/scored_candidates_train.jsonl")
    parser.add_argument("--llm_model", default="Qwen/Qwen3-8B",
                        help="Local Qwen3-8B model path for LLM_AUG generation")
    parser.add_argument("--slm_probe", default="slm_outputs/basepack_only/best_model",
                        help="Frozen G0 DeBERTa SLM probe path")
    parser.add_argument("--split", default=None,
                        help="If set, auto-sets candidates and output paths")
    args = parser.parse_args()

    if args.split:
        args.candidates = f"policy_data/dpo/candidate_hints_{args.split}.jsonl"
        args.output = f"policy_data/dpo/scored_candidates_{args.split}.jsonl"

    score_split(
        Path(args.candidates),
        Path(args.output),
        args.llm_model,
        args.slm_probe,
    )


if __name__ == "__main__":
    main()
