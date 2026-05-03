"""
03_make_candidate_hints.py
Generate 12 candidate focus_hints per event for DPO pair construction.

Candidates (12 total):
  Tier 1 - 4 template candidates (clear "what not to learn" baseline):
    h_tmpl_heuristic, h_tmpl_query, h_tmpl_deny, h_tmpl_grounding

  Tier 2 - 6 LLM-rewritten candidates from Qwen3-8B under different system prompts:
    h_llm_verifiability, h_llm_narrative, h_llm_stance, h_llm_grounding,
    h_llm_temporal, h_llm_synthesis

  Tier 3 - 2 SFT policy samples at temperature=0.7:
    h_sft_0, h_sft_1

Input:  basepack_v2/basepack_train.jsonl
        policy_outputs/sft_model  (LoRA adapter)
Output: policy_data/dpo/candidate_hints_{split}.jsonl
"""

import json
import re
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from _policy_input import build_policy_input

TEMPLATE_CANDIDATES = {
    "h_tmpl_heuristic": (
        "Focus on the central claim, supporting and refuting replies, conflict among replies, "
        "source grounding, and whether the discussion provides concrete evidence or only emotional reactions."
    ),
    "h_tmpl_query": (
        "Prioritize questioning replies and uncertainty signals. Check what details are missing "
        "from the source claim, especially time, location, actor, source attribution, and evidence form."
    ),
    "h_tmpl_deny": (
        "Prioritize replies that deny, correct, or challenge the source claim. Compare them with "
        "supportive replies and identify whether the disagreement is based on evidence or speculation."
    ),
    "h_tmpl_grounding": (
        "Focus on whether the source post contains verifiable information such as named entities, "
        "time, location, source links, or attribution. Be cautious if the claim is vague."
    ),
}

LLM_SYSTEM_PROMPTS = {
    "h_llm_verifiability": (
        "You are a rumor analysis expert. Generate a focus hint emphasizing verifiability: "
        "whether the claim can be checked against external facts, named entities, timestamps, "
        "source attribution, or official records. Be concrete and specific to this event."
    ),
    "h_llm_narrative": (
        "You are a rumor analysis expert. Generate a focus hint emphasizing narrative manipulation: "
        "emotional appeals, exaggeration, missing context, one-sided framing, or persuasion techniques "
        "in the source post and supporting replies. Be concrete and specific to this event."
    ),
    "h_llm_stance": (
        "You are a rumor analysis expert. Generate a focus hint emphasizing stance conflict: "
        "the distribution of agreement, denial, and doubt in the replies, and what this pattern "
        "reveals about the claim's credibility. Be concrete and specific to this event."
    ),
    "h_llm_grounding": (
        "You are a rumor analysis expert. Generate a focus hint emphasizing source grounding: "
        "whether the claim is anchored to a credible source, institutional authority, or direct evidence, "
        "or whether it floats without attribution. Be concrete and specific to this event."
    ),
    "h_llm_temporal": (
        "You are a rumor analysis expert. Generate a focus hint emphasizing temporal signals: "
        "timing anomalies, recirculated old content, temporal inconsistencies, or whether the claim "
        "matches the propagation timeline. Be concrete and specific to this event."
    ),
    "h_llm_synthesis": (
        "You are a rumor analysis expert. Generate a balanced, neutral focus hint that synthesizes "
        "all available signals without presuming the claim is true or fake. Identify the key uncertainty "
        "and what a downstream model should pay most attention to."
    ),
}

POLICY_INSTRUCTION = (
    "Generate a concise focus_hint for LLM-assisted rumor analysis. "
    "Return only a JSON object with the key focus_hint."
)

FALLBACK_HINT = list(TEMPLATE_CANDIDATES.values())[0]

JSON_RE = re.compile(r'\{[^{}]*"focus_hint"\s*:\s*"([^"]+)"[^{}]*\}', re.DOTALL)


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


def load_sft_policy(base_model_path: str, adapter_path: str):
    print(f"Loading SFT policy: base={base_model_path}, adapter={adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def _parse_hint(generated: str) -> str | None:
    m = JSON_RE.search(generated)
    if m:
        return m.group(1).strip()
    try:
        obj = json.loads(generated)
        if "focus_hint" in obj:
            return str(obj["focus_hint"]).strip()
    except json.JSONDecodeError:
        pass
    return None


def generate_llm_hint(tokenizer, model, system_prompt: str,
                      event: dict, max_new_tokens: int = 120) -> str:
    policy_input = build_policy_input(event)
    user_msg = (
        f"{policy_input}\n\n"
        "Generate a focus_hint (30-80 words). Return only: {\"focus_hint\": \"...\"}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return _parse_hint(generated) or FALLBACK_HINT


def generate_sft_sample(sft_tokenizer, sft_model, event: dict,
                        temperature: float = 0.7, max_new_tokens: int = 120) -> str:
    policy_input = build_policy_input(event)
    messages = [{"role": "user", "content": f"{POLICY_INSTRUCTION}\n\n{policy_input}"}]
    text = sft_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = sft_tokenizer([text], return_tensors="pt").to(sft_model.device)
    with torch.no_grad():
        output_ids = sft_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=sft_tokenizer.pad_token_id,
        )
    generated = sft_tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return _parse_hint(generated) or FALLBACK_HINT


def _release_model(model, tokenizer):
    """Delete model from GPU memory and empty CUDA cache."""
    import gc
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_split(basepack_path: Path, llm_tokenizer, llm_model,
                  sft_tokenizer, sft_model, output_path: Path,
                  sft_base_model: str = "Qwen/Qwen3-1.7B",
                  sft_adapter: str = "policy_outputs/sft_model"):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    events = []
    with open(basepack_path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    print(f"Loaded {len(events)} events from {basepack_path}")

    resume_ids: set = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    resume_ids.add(obj["event_id"])
        print(f"Resuming: {len(resume_ids)} already processed.")

    to_process = [e for e in events if e["event_id"] not in resume_ids]
    print(f"Remaining: {len(to_process)}")

    # ── Pass 1: Tier 1 (templates) + Tier 2 (LLM-rewritten) ──────────────────
    # Build partial records keyed by event_id; Tier 3 filled in Pass 2.
    partial: dict[str, dict] = {}

    with open(output_path, "a") as fout:
        for i, event in enumerate(to_process, 1):
            candidates = []

            # Tier 1: template candidates
            for hint_id, hint_text in TEMPLATE_CANDIDATES.items():
                candidates.append({"hint_id": hint_id, "focus_hint": hint_text})

            # Tier 2: LLM-rewritten candidates (Qwen3-8B)
            if llm_model is not None:
                for hint_id, system_prompt in LLM_SYSTEM_PROMPTS.items():
                    hint = generate_llm_hint(llm_tokenizer, llm_model, system_prompt, event)
                    candidates.append({"hint_id": hint_id, "focus_hint": hint})
            else:
                for hint_id in LLM_SYSTEM_PROMPTS:
                    candidates.append({"hint_id": hint_id, "focus_hint": FALLBACK_HINT})

            # Tier 3: placeholder — filled in Pass 2
            for j in range(2):
                candidates.append({"hint_id": f"h_sft_{j}", "focus_hint": FALLBACK_HINT})

            partial[event["event_id"]] = {
                "event_id": event["event_id"],
                "label": event["label"],
                "basepack_text": event["basepack_text"],
                "source_text": event.get("source_text", ""),
                "stats": event.get("stats", {}),
                "stance_dist": event.get("stance_dist", {}),
                "candidates": candidates,
                "_event": event,
            }

            if i % 50 == 0:
                print(f"  [Pass 1] Progress: {i}/{len(to_process)}")

    # ── Release Qwen3-8B before loading SFT policy ───────────────────────────
    if llm_model is not None:
        print("Releasing Qwen3-8B from GPU memory...")
        _release_model(llm_model, llm_tokenizer)
        llm_model, llm_tokenizer = None, None

    # ── Pass 2: Tier 3 (SFT policy samples at temperature=0.7) ───────────────
    need_sft = sft_model is None and Path(sft_adapter).exists()
    if need_sft:
        print("Loading SFT policy for Tier 3 candidates...")
        sft_tokenizer, sft_model = load_sft_policy(sft_base_model, sft_adapter)

    if sft_model is not None:
        for eid, rec in partial.items():
            event = rec["_event"]
            sft_hints = [generate_sft_sample(sft_tokenizer, sft_model, event) for _ in range(2)]
            for j, hint in enumerate(sft_hints):
                rec["candidates"][-(2 - j)]["focus_hint"] = hint

        if need_sft:
            print("Releasing SFT policy from GPU memory...")
            _release_model(sft_model, sft_tokenizer)

    # ── Write final records ───────────────────────────────────────────────────
    with open(output_path, "a") as fout:
        for rec in partial.values():
            out = {k: v for k, v in rec.items() if k != "_event"}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        fout.flush()

    print(f"Done. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate 12 candidate hints per event")
    parser.add_argument("--basepack", default="basepack_v2/basepack_train.jsonl")
    parser.add_argument("--llm_model", default="Qwen/Qwen3-8B",
                        help="Local Qwen3-8B model path for LLM-rewritten candidates")
    parser.add_argument("--sft_adapter", default="policy_outputs/sft_model",
                        help="SFT LoRA adapter path")
    parser.add_argument("--sft_base_model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", default="policy_data/dpo/candidate_hints_train.jsonl")
    parser.add_argument("--split", default=None,
                        help="If set, auto-sets basepack and output paths (uses basepack_v2)")
    parser.add_argument("--no_llm", action="store_true",
                        help="Skip LLM-rewritten candidates (template fallback); for testing")
    parser.add_argument("--no_sft", action="store_true",
                        help="Skip SFT policy candidates (template fallback); for testing")
    args = parser.parse_args()

    if args.split:
        args.basepack = f"basepack_v2/basepack_{args.split}.jsonl"
        args.output = f"policy_data/dpo/candidate_hints_{args.split}.jsonl"

    llm_tokenizer, llm_model = None, None
    if not args.no_llm:
        llm_tokenizer, llm_model = load_llm_8b(args.llm_model)

    sft_tokenizer, sft_model = None, None
    sft_adapter_path = Path(args.sft_adapter)
    if not args.no_sft and sft_adapter_path.exists():
        sft_tokenizer, sft_model = load_sft_policy(args.sft_base_model, args.sft_adapter)
    elif not args.no_sft:
        print(f"[WARN] SFT adapter not found at {args.sft_adapter}. Skipping SFT candidates.")

    process_split(
        Path(args.basepack),
        llm_tokenizer, llm_model,
        sft_tokenizer, sft_model,
        Path(args.output),
    )


if __name__ == "__main__":
    main()
