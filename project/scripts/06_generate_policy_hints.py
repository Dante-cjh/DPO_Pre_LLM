"""
06_generate_policy_hints.py
Run inference with a trained policy model (SFT or DPO) to generate focus_hints.

JSON parse failures fall back to the heuristic hint.
Hint length is checked to be within 30-80 English tokens (approximate).

Usage:
  # Single file mode:
  python scripts/06_generate_policy_hints.py \
      --model_path policy_outputs/sft_model \
      --input basepack_v2/basepack_test.jsonl \
      --output policy_outputs/sft_hints_test.jsonl

  # Policy mode (processes all splits):
  python scripts/06_generate_policy_hints.py --policy sft
  python scripts/06_generate_policy_hints.py --policy dpo
"""

import json
import re
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from _policy_input import build_policy_input

FALLBACK_HINT = (
    "Focus on the central claim, supporting and refuting replies, conflict among replies, "
    "source grounding, and whether the discussion provides concrete evidence or only emotional reactions."
)

INSTRUCTION = (
    "Generate a concise focus_hint for LLM-assisted rumor analysis. "
    "Return only a JSON object with the key focus_hint."
)

JSON_RE = re.compile(r'\{[^{}]*"focus_hint"\s*:\s*"([^"]+)"[^{}]*\}', re.DOTALL)

POLICY_PATHS = {
    "sft": "policy_outputs/sft_model",
    "dpo": "policy_outputs/dpo_model",
}


def load_model(base_model_path: str, adapter_path: str):
    print(f"Base model: {base_model_path}")
    print(f"Adapter:    {adapter_path}")
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


def token_count_approx(text: str) -> int:
    return len(text.split())


def constrain_hint(hint: str) -> str:
    words = hint.split()
    if len(words) > 80:
        hint = " ".join(words[:80])
    return hint


def generate_hint(tokenizer, model, event: dict, max_new_tokens: int = 200) -> str:
    policy_input = build_policy_input(event)
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n\n{policy_input}"}]
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

    hint = None
    m = JSON_RE.search(generated)
    if m:
        hint = m.group(1).strip()
    else:
        try:
            obj = json.loads(generated)
            if "focus_hint" in obj:
                hint = str(obj["focus_hint"]).strip()
        except json.JSONDecodeError:
            pass

    if not hint or token_count_approx(hint) < 5:
        return FALLBACK_HINT

    return constrain_hint(hint)


def process_file(basepack_path: Path, output_path: Path,
                 tokenizer, model, max_new_tokens: int = 200):
    events = []
    with open(basepack_path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    resume_ids: set = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    resume_ids.add(obj["event_id"])
        print(f"  Resuming: {len(resume_ids)} done.")

    to_process = [e for e in events if e["event_id"] not in resume_ids]
    print(f"  Processing {len(to_process)} events from {basepack_path}")

    fallback_count = 0
    with open(output_path, "a") as fout:
        for i, event in enumerate(to_process, 1):
            hint = generate_hint(tokenizer, model, event, max_new_tokens)
            if hint == FALLBACK_HINT:
                fallback_count += 1

            record = {"event_id": event["event_id"], "focus_hint": hint}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if i % 50 == 0:
                print(f"    Progress: {i}/{len(to_process)} | fallbacks: {fallback_count}")

    print(f"  Done. Fallbacks: {fallback_count}/{len(to_process)}. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate policy hints for all splits")
    parser.add_argument("--model_path", default=None,
                        help="Path to LoRA adapter (single-file mode)")
    parser.add_argument("--input", default=None,
                        help="BasePack file (single-file mode)")
    parser.add_argument("--output", default=None,
                        help="Output file (single-file mode)")
    parser.add_argument("--base_model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--policy", choices=["sft", "dpo"], default=None,
                        help="Policy type — processes all splits automatically")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    if args.policy:
        adapter_path = POLICY_PATHS[args.policy]
        tokenizer, model = load_model(args.base_model, adapter_path)
        for split in args.splits:
            basepack_path = Path(f"basepack_v2/basepack_{split}.jsonl")
            output_path = Path(f"policy_outputs/{args.policy}_hints_{split}.jsonl")
            if not basepack_path.exists():
                print(f"[WARN] Not found: {basepack_path}, skipping.")
                continue
            print(f"\n=== {args.policy} / {split} ===")
            process_file(basepack_path, output_path, tokenizer, model, args.max_new_tokens)
    else:
        if not args.model_path or not args.input or not args.output:
            raise ValueError("Provide --model_path, --input, --output OR use --policy")
        tokenizer, model = load_model(args.base_model, args.model_path)
        process_file(Path(args.input), Path(args.output), tokenizer, model, args.max_new_tokens)


if __name__ == "__main__":
    main()
