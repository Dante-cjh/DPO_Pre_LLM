"""
07_run_qwen_turbo_aug.py
Generate LLM_AUG for each split by calling Qwen3-turbo with the advisor scaffold.

Three call modes:
  --method heuristic   : uses fixed heuristic_hint.txt for every event
  --method sft_hint    : reads per-event hints from policy_outputs/sft_hints_{split}.jsonl
  --method dpo_hint    : reads per-event hints from policy_outputs/dpo_hints_{split}.jsonl

The scaffold asks for 6 analysis fields only (no weak_label / weak_confidence).

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

import yaml
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REQUIRED_FIELDS = {
    "claim_summary", "supporting_signals", "refuting_signals",
    "conflict_summary", "verification_focus", "uncertainty_note",
}

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

HINT_FILE_MAP = {
    "sft_hint": "policy_outputs/sft_hints_{split}.jsonl",
    "dpo_hint": "policy_outputs/dpo_hints_{split}.jsonl",
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    env_key = (
        os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
    )
    if env_key:
        cfg["api"]["api_key"] = env_key
    if not cfg["api"].get("api_key"):
        raise EnvironmentError("API key not set. Export DASHSCOPE_API_KEY.")
    return cfg


def load_scaffold(path: str) -> str:
    with open(path) as f:
        return f.read()


def load_heuristic_hint(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


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


def render_prompt(scaffold: str, focus_hint: str, basepack_text: str) -> str:
    return scaffold.replace("{focus_hint}", focus_hint).replace("{basepack_text}", basepack_text)


def parse_aug_response(raw: str) -> dict | None:
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


def call_api(client, model: str, prompt: str,
             max_retries: int, retry_delay: int) -> tuple[str | None, int, int]:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
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


def run_aug(basepack_path: Path, scaffold_path: str, hint_source: str,
            output_path: Path, config_path: str, fallback_hint: str):
    cfg = load_config(config_path)
    api_cfg = cfg["api"]
    client = OpenAI(api_key=api_cfg["api_key"], base_url=api_cfg.get("base_url"))
    model = api_cfg["model"]
    max_retries = api_cfg.get("max_retries", 3)
    retry_delay = api_cfg.get("retry_delay", 5)

    scaffold = load_scaffold(scaffold_path)

    events = []
    with open(basepack_path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    hint_map: dict[str, str] | None = None
    fixed_hint: str | None = None

    if hint_source == "heuristic":
        fixed_hint = fallback_hint
    else:
        hint_map = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    resume_ids: set = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    resume_ids.add(obj["event_id"])
        print(f"Resuming: {len(resume_ids)} already done.")

    to_process = [e for e in events if e["event_id"] not in resume_ids]
    print(f"Processing {len(to_process)} events | model={model}")

    success = 0
    parse_fail = 0
    total_in_tok = 0
    total_out_tok = 0

    with open(output_path, "a") as fout:
        for i, event in enumerate(to_process, 1):
            eid = event["event_id"]

            if fixed_hint is not None:
                hint = fixed_hint
            else:
                hint = hint_map.get(eid, fallback_hint) if hint_map is not None else fallback_hint

            prompt = render_prompt(scaffold, hint, event["basepack_text"])
            raw, in_tok, out_tok = call_api(client, model, prompt, max_retries, retry_delay)
            total_in_tok += in_tok
            total_out_tok += out_tok

            if raw is None:
                parsed = None
                json_valid = 0
            else:
                parsed = parse_aug_response(raw)
                json_valid = 1 if parsed is not None else 0

            if parsed is not None:
                success += 1
            else:
                parse_fail += 1
                if raw:
                    print(f"  [PARSE FAIL] {eid} | raw: {(raw or '')[:80]}")

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
                print(f"  Progress: {i}/{len(to_process)} | "
                      f"success={success} | fail={parse_fail} | "
                      f"avg_out_tok={total_out_tok/n:.0f}")

    n = max(success, 1)
    print(f"\nDone. success={success}, parse_fail={parse_fail}")
    print(f"  avg_in_tok={total_in_tok/n:.0f}, avg_out_tok={total_out_tok/n:.0f}")
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-turbo LLM augmentation")
    parser.add_argument("--basepack", default=None,
                        help="BasePack file path (required in single mode)")
    parser.add_argument("--method", choices=["heuristic", "sft_hint", "dpo_hint"],
                        default="heuristic")
    parser.add_argument("--hint", default="prompts/heuristic_hint.txt",
                        help="Heuristic hint file (used when method=heuristic)")
    parser.add_argument("--hint_file", default=None,
                        help="Per-event hint jsonl (used when method=sft_hint/dpo_hint)")
    parser.add_argument("--scaffold", default="prompts/advisor_scaffold.txt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="configs/api_config.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    fallback_hint = load_heuristic_hint(args.hint)

    def get_hint_source_and_map(method, split, hint_file_arg):
        if method == "heuristic":
            return "heuristic", None
        if hint_file_arg:
            return method, hint_file_arg
        tmpl = HINT_FILE_MAP.get(method, "")
        return method, tmpl.replace("{split}", split)

    if args.basepack is not None and args.output is not None:
        split = "custom"
        hint_source, hint_file = get_hint_source_and_map(args.method, split, args.hint_file)
        hint_map = {}
        if hint_file:
            hint_map = load_hint_map(hint_file)

        cfg = load_config(args.config)
        fallback = fallback_hint

        class _Args:
            pass

        run_aug(Path(args.basepack), args.scaffold, hint_source,
                Path(args.output), args.config, fallback)
        return

    for split in args.splits:
        basepack_path = Path(f"basepack/basepack_{split}.jsonl")
        if not basepack_path.exists():
            print(f"[WARN] Not found: {basepack_path}, skipping.")
            continue

        hint_source, hint_file = get_hint_source_and_map(args.method, split, args.hint_file)
        output_path = Path(f"llm_aug/{args.method}/{split}.jsonl")

        hint_map_loaded: dict[str, str] = {}
        if hint_file and args.method != "heuristic":
            hint_map_loaded = load_hint_map(hint_file)

        cfg = load_config(args.config)
        api_cfg = cfg["api"]
        client = OpenAI(api_key=api_cfg["api_key"], base_url=api_cfg.get("base_url"))
        model = api_cfg["model"]
        max_retries = api_cfg.get("max_retries", 3)
        retry_delay = api_cfg.get("retry_delay", 5)
        scaffold = load_scaffold(args.scaffold)

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
            print(f"[{split}] Resuming: {len(resume_ids)} done.")

        to_process = [e for e in events if e["event_id"] not in resume_ids]
        print(f"\n=== method={args.method} / split={split} | {len(to_process)} events ===")

        success = 0
        parse_fail = 0
        total_out_tok = 0

        with open(output_path, "a") as fout:
            for i, event in enumerate(to_process, 1):
                eid = event["event_id"]

                if args.method == "heuristic":
                    hint = fallback_hint
                else:
                    hint = hint_map_loaded.get(eid, fallback_hint)

                prompt = render_prompt(scaffold, hint, event["basepack_text"])
                raw, in_tok, out_tok = call_api(client, model, prompt, max_retries, retry_delay)
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


if __name__ == "__main__":
    main()
