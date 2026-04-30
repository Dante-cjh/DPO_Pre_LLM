"""
03_make_candidate_hints.py
Generate 6 candidate focus_hints per event for DPO pair construction.

Candidates:
  h0_heuristic  - fixed heuristic hint
  h1_sft        - SFT policy model output
  h2_query      - query_focus template
  h3_deny       - deny_focus template
  h4_grounding  - source_grounding_focus template
  h5_conflict   - conflict_propagation_focus template

Input:  basepack/basepack_train.jsonl  (or val)
        policy_outputs/sft_model       (LoRA adapter)
Output: policy_data/dpo/candidate_hints_{split}.jsonl
"""

import json
import re
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

HEURISTIC_HINT = (
    "Focus on the central claim, supporting and refuting replies, conflict among replies, "
    "source grounding, and whether the discussion provides concrete evidence or only emotional reactions."
)

HINT_TEMPLATES = {
    "h2_query": (
        "Prioritize questioning replies and uncertainty signals. Check what details are missing "
        "from the source claim, especially time, location, actor, source attribution, and evidence form."
    ),
    "h3_deny": (
        "Prioritize replies that deny, correct, or challenge the source claim. Compare them with "
        "supportive replies and identify whether the disagreement is based on evidence or speculation."
    ),
    "h4_grounding": (
        "Focus on whether the source post contains verifiable information such as named entities, "
        "time, location, source links, or attribution. Be cautious if the claim is vague."
    ),
    "h5_conflict": (
        "Focus on conflict and propagation signals. Check whether replies show agreement, denial, "
        "questions, repeated claims, or attention-driven reactions without concrete evidence."
    ),
}

INSTRUCTION = (
    "Generate a concise focus_hint for LLM-assisted rumor analysis. "
    "Return only a JSON object with the key focus_hint."
)

JSON_RE = re.compile(r'\{[^{}]*"focus_hint"\s*:\s*"([^"]+)"[^{}]*\}', re.DOTALL)


def build_policy_input(event: dict) -> str:
    lines = ["SOURCE:", event.get("source_text", ""), "", "REPLIES:"]
    for i, reply in enumerate(event.get("selected_replies", []), 1):
        lines.append(f"{i}. {reply}")
    stats = event.get("stats", {})
    lines.append("")
    lines.append(
        f"STATS:\nreply_count={stats.get('num_replies', 0)} | "
        f"depth={stats.get('max_depth', 0)} | "
        f"branches={stats.get('num_branches', 0)}"
    )
    return "\n".join(lines)


def load_model(base_model_path: str, adapter_path: str):
    print(f"Loading base model: {base_model_path}")
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
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_hint(tokenizer, model, event: dict, max_new_tokens: int = 200) -> str | None:
    policy_input = build_policy_input(event)
    messages = [
        {"role": "user", "content": f"{INSTRUCTION}\n\n{policy_input}"}
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


def process_split(basepack_path: Path, adapter_path: str, base_model: str,
                  output_path: Path, batch_size: int):
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

    tokenizer, model = load_model(base_model, adapter_path)

    with open(output_path, "a") as fout:
        for i, event in enumerate(to_process, 1):
            sft_hint = generate_hint(tokenizer, model, event)
            if sft_hint is None:
                sft_hint = HEURISTIC_HINT

            candidates = [
                {"hint_id": "h0_heuristic", "focus_hint": HEURISTIC_HINT},
                {"hint_id": "h1_sft", "focus_hint": sft_hint},
            ]
            for hint_id, hint_text in HINT_TEMPLATES.items():
                candidates.append({"hint_id": hint_id, "focus_hint": hint_text})

            record = {
                "event_id": event["event_id"],
                "label": event["label"],
                "basepack_text": event["basepack_text"],
                "candidates": candidates,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if i % 50 == 0:
                print(f"  Progress: {i}/{len(to_process)}")

    print(f"Done. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate candidate hints for DPO")
    parser.add_argument("--basepack", default="basepack/basepack_train.jsonl")
    parser.add_argument("--adapter_path", default="policy_outputs/sft_model")
    parser.add_argument("--base_model", default="Qwen/Qwen3-1.7B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output", default="policy_data/dpo/candidate_hints_train.jsonl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split", default=None,
                        help="If set (train/val), auto-sets basepack and output paths")
    args = parser.parse_args()

    if args.split:
        args.basepack = f"basepack/basepack_{args.split}.jsonl"
        args.output = f"policy_data/dpo/candidate_hints_{args.split}.jsonl"

    process_split(
        Path(args.basepack),
        args.adapter_path,
        args.base_model,
        Path(args.output),
        args.batch_size,
    )


if __name__ == "__main__":
    main()
