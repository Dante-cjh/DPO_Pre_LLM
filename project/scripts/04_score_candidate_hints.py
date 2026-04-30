"""
04_score_candidate_hints.py
Score each candidate hint by calling Qwen3-turbo with the scoring scaffold.

For each (event, hint) pair, the LLM outputs an 8-field JSON including
weak_label and weak_confidence, which are used to compute reward for DPO.

Input:  policy_data/dpo/candidate_hints_{split}.jsonl
        prompts/scoring_scaffold.txt
Output: policy_data/dpo/scored_candidates_{split}.jsonl
"""

import json
import math
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
    "weak_label", "weak_confidence",
}

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def load_config(config_path: str = "configs/api_config.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    env_key = (
        os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
    )
    if env_key:
        cfg["api"]["api_key"] = env_key
    if not cfg["api"].get("api_key"):
        raise EnvironmentError(
            "API key not set. Export DASHSCOPE_API_KEY or set in .env"
        )
    return cfg


def load_scaffold(path: str = "prompts/scoring_scaffold.txt") -> str:
    with open(path) as f:
        return f.read()


def render_prompt(scaffold: str, focus_hint: str, basepack_text: str) -> str:
    return scaffold.replace("{focus_hint}", focus_hint).replace("{basepack_text}", basepack_text)


def parse_response(raw: str) -> dict | None:
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

    weak_label = str(obj.get("weak_label", "")).lower().strip()
    if weak_label not in ("true", "fake"):
        return None

    try:
        conf = float(obj["weak_confidence"])
    except (ValueError, TypeError):
        return None

    obj["weak_label"] = weak_label
    obj["weak_confidence"] = max(0.0, min(1.0, conf))
    return obj


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


def score_split(candidate_path: Path, scaffold_path: str, output_path: Path,
                config_path: str):
    cfg = load_config(config_path)
    api_cfg = cfg["api"]
    client = OpenAI(api_key=api_cfg["api_key"], base_url=api_cfg.get("base_url"))
    model = api_cfg["model"]
    max_retries = api_cfg.get("max_retries", 3)
    retry_delay = api_cfg.get("retry_delay", 5)
    scaffold = load_scaffold(scaffold_path)

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
    success = 0
    parse_fail = 0

    with open(output_path, "a") as fout:
        for event in events:
            eid = event["event_id"]
            for cand in event["candidates"]:
                hint_id = cand["hint_id"]
                key = (eid, hint_id)
                if key in resume_keys:
                    done += 1
                    continue

                prompt = render_prompt(scaffold, cand["focus_hint"], event["basepack_text"])
                raw, in_tok, out_tok = call_api(client, model, prompt, max_retries, retry_delay)

                if raw is None:
                    parsed = None
                    json_valid = 0
                else:
                    parsed = parse_response(raw)
                    json_valid = 1 if parsed is not None else 0

                if parsed is None:
                    parse_fail += 1
                    record = {
                        "event_id": eid,
                        "hint_id": hint_id,
                        "focus_hint": cand["focus_hint"],
                        "gold_label": event["label"],
                        "llm_json_valid": json_valid,
                        "weak_label": None,
                        "weak_confidence": None,
                        "analysis": None,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                    }
                else:
                    success += 1
                    analysis = {k: parsed[k] for k in (
                        "claim_summary", "supporting_signals", "refuting_signals",
                        "conflict_summary", "verification_focus", "uncertainty_note"
                    )}
                    record = {
                        "event_id": eid,
                        "hint_id": hint_id,
                        "focus_hint": cand["focus_hint"],
                        "gold_label": event["label"],
                        "llm_json_valid": 1,
                        "weak_label": parsed["weak_label"],
                        "weak_confidence": parsed["weak_confidence"],
                        "analysis": analysis,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                    }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                done += 1

                if done % 100 == 0:
                    print(f"  Progress: {done}/{total} | success={success} | fail={parse_fail}")

    print(f"\nDone. success={success}, parse_fail={parse_fail}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Score candidate hints via Qwen3-turbo")
    parser.add_argument("--candidates", default="policy_data/dpo/candidate_hints_train.jsonl")
    parser.add_argument("--scaffold", default="prompts/scoring_scaffold.txt")
    parser.add_argument("--output", default="policy_data/dpo/scored_candidates_train.jsonl")
    parser.add_argument("--config", default="configs/api_config.yaml")
    parser.add_argument("--split", default=None,
                        help="If set, auto-sets candidates and output paths")
    args = parser.parse_args()

    if args.split:
        args.candidates = f"policy_data/dpo/candidate_hints_{args.split}.jsonl"
        args.output = f"policy_data/dpo/scored_candidates_{args.split}.jsonl"

    score_split(
        Path(args.candidates), args.scaffold, Path(args.output), args.config
    )


if __name__ == "__main__":
    main()
