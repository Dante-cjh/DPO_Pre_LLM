"""
07_run_qwen_turbo_aug.py
Generate LLM_AUG for each split using local Qwen3-8B or an OpenAI-compatible API.

Three call modes:
  --method heuristic   : uses fixed heuristic_hint.txt for every event
  --method sft_hint    : reads per-event hints from policy_outputs/sft_hints_{split}.jsonl
  --method dpo_hint    : reads per-event hints from policy_outputs/dpo_hints_{split}.jsonl

Inference backend:
  --backend api        : OpenAI-compatible API with concurrent workers (DEFAULT, recommended)
                         Works with vLLM local server or any remote API.
  --backend local      : local Qwen3-8B HuggingFace inference (slow, for offline use)

API endpoint override (backend=api):
  --api_base_url       : override base_url in api_config.yaml  (e.g. http://localhost:8000/v1)
  --api_model          : override model name                    (e.g. Qwen3-8B)
  --api_key            : override api key                       (e.g. EMPTY for vLLM)
  --api_workers N      : number of concurrent requests          (default: 50)

Output per line:
{
  "event_id": str,
  "label": int,
  "llm_aug": {...} | null,
  "json_valid": int,
  "raw_response": str | null
}
"""

import json
import os
import re
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    "Only use information present in the event. Do not invent facts. "
    "Do NOT include weak_label or weak_confidence."
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


# ── response parsing ──────────────────────────────────────────────────────────

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


# ── API backend (concurrent) ──────────────────────────────────────────────────

def _api_call_one(client: "OpenAI", model: str,
                  focus_hint: str, basepack_text: str,
                  max_retries: int, retry_delay: int) -> tuple[str | None, int, int]:
    user_msg = f"[FOCUS_HINT]\n{focus_hint}\n\n[EVENT]\n{basepack_text}"
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
            usage = resp.usage
            in_tok  = usage.prompt_tokens     if usage else 0
            out_tok = usage.completion_tokens if usage else len(raw.split())
            return raw, in_tok, out_tok
        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                print(f"    [API error after {max_retries} retries] {e}")
    return None, 0, 0


def run_split_api(events: list, hint_map: dict, fixed_hint: str | None,
                  output_path: Path,
                  client: "OpenAI", model: str,
                  api_workers: int, max_retries: int, retry_delay: int):
    """Process all pending events concurrently via API."""

    resume_ids = _load_resume_ids(output_path)
    to_process = [e for e in events if e["event_id"] not in resume_ids]
    print(f"  Processing {len(to_process)} events | backend=api | workers={api_workers}")
    if not to_process:
        print("  Nothing to do.")
        return

    total = len(to_process)
    success = 0
    parse_fail = 0
    t0 = time.time()
    write_lock = threading.Lock()

    def _task(event: dict) -> dict:
        eid = event["event_id"]
        hint = fixed_hint if fixed_hint else hint_map.get(eid, FALLBACK_HINT)
        raw, _, _ = _api_call_one(client, model, hint, event["basepack_text"],
                                   max_retries, retry_delay)
        parsed = parse_aug_response(raw) if raw else None
        return {
            "event_id":    eid,
            "label":       event["label"],
            "llm_aug":     parsed,
            "json_valid":  1 if parsed is not None else 0,
            "raw_response": raw,
        }

    completed = 0
    with open(output_path, "a") as fout, \
         ThreadPoolExecutor(max_workers=api_workers) as executor:
        futures = {executor.submit(_task, e): e for e in to_process}
        for fut in as_completed(futures):
            record = fut.result()
            with write_lock:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
            completed += 1
            if record["json_valid"]:
                success += 1
            else:
                parse_fail += 1
            if completed % 50 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed * 3600 if elapsed > 0 else 0
                print(f"  {completed}/{total} | success={success} | "
                      f"fail={parse_fail} | rate={rate:.0f} ev/hr")

    print(f"  Done: success={success}, fail={parse_fail}. Output: {output_path}")


# ── local backend (sequential) ────────────────────────────────────────────────

def load_local_model(model_name: str):
    print(f"Loading local LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def _call_local(tokenizer, model, focus_hint: str, basepack_text: str,
                max_new_tokens: int = 300) -> tuple[str | None, int, int]:
    user_msg = f"[FOCUS_HINT]\n{focus_hint}\n\n[EVENT]\n{basepack_text}"
    messages = [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    in_tok = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    out_tok = output_ids.shape[1] - in_tok
    raw = tokenizer.decode(output_ids[0][in_tok:], skip_special_tokens=True).strip()
    return raw, in_tok, out_tok


def run_split_local(events: list, hint_map: dict, fixed_hint: str | None,
                    output_path: Path, tokenizer, model):
    resume_ids = _load_resume_ids(output_path)
    to_process = [e for e in events if e["event_id"] not in resume_ids]
    print(f"  Processing {len(to_process)} events | backend=local")

    success = parse_fail = 0
    with open(output_path, "a") as fout:
        for i, event in enumerate(to_process, 1):
            eid = event["event_id"]
            hint = fixed_hint if fixed_hint else hint_map.get(eid, FALLBACK_HINT)
            raw, _, _ = _call_local(tokenizer, model, hint, event["basepack_text"])
            parsed = parse_aug_response(raw) if raw else None
            record = {
                "event_id":    eid,
                "label":       event["label"],
                "llm_aug":     parsed,
                "json_valid":  1 if parsed is not None else 0,
                "raw_response": raw,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            if parsed:
                success += 1
            else:
                parse_fail += 1
            if i % 50 == 0:
                print(f"  {i}/{len(to_process)} | success={success} | fail={parse_fail}")

    print(f"  Done: success={success}, fail={parse_fail}. Output: {output_path}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_resume_ids(output_path: Path) -> set:
    ids: set = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    ids.add(json.loads(line)["event_id"])
        if ids:
            print(f"  Resuming: {len(ids)} events already done.")
    return ids


def load_hint_map(hint_file: str) -> dict[str, str]:
    path = Path(hint_file)
    if not path.exists():
        raise FileNotFoundError(f"Hint file not found: {hint_file}")
    hint_map: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                hint_map[obj["event_id"]] = obj["focus_hint"]
    return hint_map


def load_heuristic_hint(path: str) -> str:
    p = Path(path)
    return p.read_text().strip() if p.exists() else FALLBACK_HINT


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LLM augmentation")
    parser.add_argument("--method", choices=["heuristic", "sft_hint", "dpo_hint"],
                        default="heuristic")
    parser.add_argument("--backend", choices=["api", "local"], default="api")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--basepack_dir", default="basepack_v2")
    parser.add_argument("--hint", default="prompts/heuristic_hint.txt")
    parser.add_argument("--hint_file", default=None)

    # API options
    parser.add_argument("--config", default="configs/api_config.yaml")
    parser.add_argument("--api_base_url", default=None,
                        help="Override base_url in config (e.g. http://localhost:8000/v1)")
    parser.add_argument("--api_model", default=None,
                        help="Override model name in config (e.g. Qwen3-8B)")
    parser.add_argument("--api_key", default=None,
                        help="Override API key (use 'EMPTY' for local vLLM)")
    parser.add_argument("--api_workers", type=int, default=50,
                        help="Concurrent API workers (backend=api)")
    parser.add_argument("--api_max_retries", type=int, default=3)
    parser.add_argument("--api_retry_delay", type=int, default=5)

    # Local options
    parser.add_argument("--llm_model", default="Qwen/Qwen3-8B")

    args = parser.parse_args()

    heuristic_hint = load_heuristic_hint(args.hint)

    # ── set up backend ────────────────────────────────────────────────────────
    api_client = api_model_name = None
    api_max_retries = args.api_max_retries
    api_retry_delay = args.api_retry_delay
    local_tokenizer = local_model = None

    if args.backend == "api":
        if not _OPENAI_AVAILABLE:
            raise ImportError("pip install openai pyyaml")
        # Load yaml config, then apply CLI overrides
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        api_cfg = cfg.get("api", {})

        base_url = args.api_base_url or api_cfg.get("base_url")
        model_name = args.api_model or api_cfg.get("model", "Qwen3-8B")
        api_key = (
            args.api_key
            or os.environ.get("DASHSCOPE_API_KEY", "").strip()
            or os.environ.get("LLM_API_KEY", "").strip()
            or api_cfg.get("api_key", "EMPTY")
        )
        api_max_retries = api_cfg.get("max_retries", args.api_max_retries)
        api_retry_delay = api_cfg.get("retry_delay", args.api_retry_delay)

        api_client = OpenAI(api_key=api_key, base_url=base_url)
        api_model_name = model_name
        print(f"Backend: api | base_url={base_url} | model={model_name} | workers={args.api_workers}")
    else:
        local_tokenizer, local_model = load_local_model(args.llm_model)

    # ── process each split ────────────────────────────────────────────────────
    for split in args.splits:
        basepack_path = Path(args.basepack_dir) / f"basepack_{split}.jsonl"
        if not basepack_path.exists():
            print(f"[WARN] Not found: {basepack_path}, skipping.")
            continue

        # Resolve hint source
        if args.method == "heuristic":
            fixed_hint = heuristic_hint
            hint_map: dict[str, str] = {}
        else:
            fixed_hint = None
            hint_file = args.hint_file or HINT_FILE_MAP.get(args.method, "").replace("{split}", split)
            hint_map = load_hint_map(hint_file) if hint_file else {}

        output_path = Path(f"llm_aug/{args.method}/{split}.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n=== method={args.method} / split={split} ===")

        events = []
        with open(basepack_path) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

        if args.backend == "api":
            run_split_api(
                events, hint_map, fixed_hint, output_path,
                api_client, api_model_name,
                args.api_workers, api_max_retries, api_retry_delay,
            )
        else:
            run_split_local(events, hint_map, fixed_hint, output_path,
                            local_tokenizer, local_model)


if __name__ == "__main__":
    main()
