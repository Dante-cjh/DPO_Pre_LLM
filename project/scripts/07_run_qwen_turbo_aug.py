"""
07_run_qwen_turbo_aug.py
Generate LLM_AUG for each split using local Qwen3-8B (advisor scaffold).

Three call modes:
  --method heuristic   : uses fixed heuristic_hint.txt for every event
  --method sft_hint    : reads per-event hints from policy_outputs/sft_hints_{split}.jsonl
  --method dpo_hint    : reads per-event hints from policy_outputs/dpo_hints_{split}.jsonl

Inference backend:
  --backend local      : local Qwen3-8B HuggingFace model (default)
  --backend api        : OpenAI-compatible API (legacy, kept for ablations)

Output per line:
{
  "event_id": str,
  "label": int,
  "llm_aug": {
    "claim_summary": str,
    "supporting_signals": [str],
    "refuting_signals": [str],
    "conflict_summary": str,
    "verification_focus": [str],
    "uncertainty_note": str
  } | null,
  "json_valid": int,
  "raw_response": str | null
}
"""

import json
import os
import re
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import yaml
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REQUIRED_FIELDS = {
    "claim_summary", "supporting_signals", "refuting_signals",
    "conflict_summary", "verification_focus", "uncertainty_note",
}

ADVISOR_SYSTEM_PROMPT = (
    "You are a misinformation analysis assistant. Analyze the given social media event and "
    "follow the provided focus hint. Return a valid JSON object with exactly these fields:\n"
    "claim_summary, supporting_signals (list), refuting_signals (list), conflict_summary, "
    "verification_focus (list), uncertainty_note.\n"
    "Only use information present in the event. Do not invent facts. Do NOT include weak_label or weak_confidence."
)

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

HINT_FILE_MAP = {
    "sft_hint": "policy_outputs/sft_hints_{split}.jsonl",
    "dpo_hint": "policy_outputs/dpo_hints_{split}.jsonl",
}

FALLBACK_HINT = (
    "Focus on the central claim, supporting and refuting replies, conflict among replies, "
    "source grounding, and whether the discussion provides concrete evidence or only emotional reactions."
)


def load_local_model(model_name: str):
    print(f"Loading local LLM: {model_name}")
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


def call_local(tokenizer, model, focus_hint: str, basepack_text: str,
               max_new_tokens: int = 300) -> tuple[str | None, int, int]:
    user_msg = (
        f"[FOCUS_HINT]\n{focus_hint}\n\n"
        f"[EVENT]\n{basepack_text}"
    )
    messages = [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    in_tok = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    out_tok = output_ids.shape[1] - in_tok
    raw = tokenizer.decode(
        output_ids[0][in_tok:], skip_special_tokens=True
    ).strip()
    return raw, in_tok, out_tok


def call_api(client, model: str, focus_hint: str, basepack_text: str,
             max_retries: int, retry_delay: int) -> tuple[str | None, int, int]:
    user_msg = (
        f"[FOCUS_HINT]\n{focus_hint}\n\n"
        f"[EVENT]\n{basepack_text}"
    )
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
            usage = resp.usage
            in_tok = usage.prompt_tokens if usage else 0
            out_tok = usage.completion_tokens if usage else 0
            return raw, in_tok, out_tok
        except Exception as e:
            print(f"    [attempt {attempt}/{max_retries}] API error: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
    return None, 0, 0


def parse_aug_response(raw: str) -> dict | None:
    if not raw:
        return None
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
    if not REQUIRED_FIELDS.issubset(obj.keys()):
        return None
    for bad in ("weak_label", "weak_confidence"):
        obj.pop(bad, None)
    return {k: obj[k] for k in REQUIRED_FIELDS}


def load_hint_map(hint_file: str) -> dict[str, str]:
    hint_map: dict[str, str] = {}
    path = Path(hint_file)
    if not path.exists():
        raise FileNotFoundError(f"Hint file not found: {hint_file}")
    with open(path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                hint_map[obj["event_id"]] = obj["focus_hint"]
    return hint_map


def load_heuristic_hint(path: str) -> str:
    p = Path(path)
    if p.exists():
        return p.read_text().strip()
    return FALLBACK_HINT


def run_split(basepack_path: Path, method: str, hint_source: str | None,
              output_path: Path, backend: str,
              local_tokenizer=None, local_model=None,
              api_client=None, api_model: str = "",
              api_max_retries: int = 3, api_retry_delay: int = 5):
    events = []
    with open(basepack_path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    hint_map: dict[str, str] = {}
    fixed_hint: str | None = None

    if method == "heuristic":
        fixed_hint = hint_source or FALLBACK_HINT
    elif hint_source:
        hint_map = load_hint_map(hint_source)

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
    print(f"  Processing {len(to_process)} events | backend={backend}")

    success = 0
    parse_fail = 0
    total_out_tok = 0

    with open(output_path, "a") as fout:
        for i, event in enumerate(to_process, 1):
            eid = event["event_id"]
            hint = fixed_hint if fixed_hint else hint_map.get(eid, FALLBACK_HINT)

            if backend == "local":
                raw, in_tok, out_tok = call_local(
                    local_tokenizer, local_model, hint, event["basepack_text"]
                )
            else:
                raw, in_tok, out_tok = call_api(
                    api_client, api_model, hint, event["basepack_text"],
                    api_max_retries, api_retry_delay
                )

            total_out_tok += out_tok
            parsed = parse_aug_response(raw) if raw else None
            json_valid = 1 if parsed is not None else 0

            if parsed is not None:
                success += 1
            else:
                parse_fail += 1

            record = {
                "event_id": eid,
                "label": event["label"],
                "llm_aug": parsed,
                "json_valid": json_valid,
                "raw_response": raw,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if i % 50 == 0:
                n = max(success, 1)
                print(f"  {i}/{len(to_process)} | success={success} | "
                      f"fail={parse_fail} | avg_out_tok={total_out_tok/n:.0f}")

    print(f"  Done: success={success}, fail={parse_fail}. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM augmentation (local Qwen3-8B or API)")
    parser.add_argument("--method", choices=["heuristic", "sft_hint", "dpo_hint"],
                        default="heuristic")
    parser.add_argument("--backend", choices=["local", "api"], default="local",
                        help="Inference backend: local Qwen3-8B or OpenAI-compatible API")
    parser.add_argument("--llm_model", default="Qwen/Qwen3-8B",
                        help="Local model path (used when backend=local)")
    parser.add_argument("--hint", default="prompts/heuristic_hint.txt",
                        help="Heuristic hint file (method=heuristic)")
    parser.add_argument("--hint_file", default=None,
                        help="Per-event hint jsonl (method=sft_hint/dpo_hint, single split)")
    parser.add_argument("--config", default="configs/api_config.yaml",
                        help="API config (backend=api only)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--basepack_dir", default="basepack_v2",
                        help="Directory with basepack_v2 files")
    args = parser.parse_args()

    heuristic_hint = load_heuristic_hint(args.hint)

    local_tokenizer, local_model = None, None
    api_client, api_model_name = None, ""
    api_max_retries, api_retry_delay = 3, 5

    if args.backend == "local":
        local_tokenizer, local_model = load_local_model(args.llm_model)
    else:
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai and pyyaml required for API backend")
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        api_cfg = cfg["api"]
        env_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if env_key:
            api_cfg["api_key"] = env_key
        api_client = OpenAI(api_key=api_cfg["api_key"], base_url=api_cfg.get("base_url"))
        api_model_name = api_cfg["model"]
        api_max_retries = api_cfg.get("max_retries", 3)
        api_retry_delay = api_cfg.get("retry_delay", 5)

    for split in args.splits:
        basepack_path = Path(args.basepack_dir) / f"basepack_{split}.jsonl"
        if not basepack_path.exists():
            print(f"[WARN] Not found: {basepack_path}, skipping.")
            continue

        if args.method == "heuristic":
            hint_source = heuristic_hint
        elif args.hint_file:
            hint_source = args.hint_file
        else:
            tmpl = HINT_FILE_MAP.get(args.method, "")
            hint_source = tmpl.replace("{split}", split) if tmpl else None

        output_path = Path(f"llm_aug/{args.method}/{split}.jsonl")
        print(f"\n=== method={args.method} / split={split} ===")

        run_split(
            basepack_path, args.method, hint_source, output_path,
            args.backend,
            local_tokenizer, local_model,
            api_client, api_model_name, api_max_retries, api_retry_delay,
        )


if __name__ == "__main__":
    main()
