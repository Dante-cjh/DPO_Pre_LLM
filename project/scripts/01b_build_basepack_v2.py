"""
01b_build_basepack_v2.py
Build BasePack-v2-Stance text representations from raw split files.

Differences from v1:
  - Reply count raised to 15-20 (max 4-5 per stance bucket)
  - Stance classification via zero-shot Qwen3-1.7B
  - Replies formatted as [REPLY-{STANCE}_N] blocks
  - [STANCE_DIST] appended to stats block

Output: basepack_v2/basepack_{split}.jsonl
Each line:
{
  "event_id": str,
  "topic": str,
  "label": int,
  "basepack_text": str,
  "source_text": str,
  "selected_replies": [{"text": str, "stance": str}],
  "stats": {"num_replies": int, "max_depth": int, "num_branches": int, "time_span": str},
  "stance_dist": {"support": float, "deny": float, "query": float, "neutral": float}
}
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

STANCES = ["support", "deny", "query", "neutral"]
MAX_PER_BUCKET = 5
MAX_TOTAL_REPLIES = 20

STANCE_SYSTEM_PROMPT = (
    "You are a stance classifier. Given a source claim and a reply, classify the reply's stance "
    "toward the claim as one of: support, deny, query, neutral.\n"
    "- support: the reply agrees with or endorses the claim\n"
    "- deny: the reply challenges, contradicts, or refutes the claim\n"
    "- query: the reply asks questions or expresses doubt without clearly supporting or denying\n"
    "- neutral: the reply is off-topic, ambiguous, or unrelated to the claim's veracity\n"
    "Output exactly one word: support, deny, query, or neutral."
)


def load_stance_model(model_name: str):
    print(f"Loading stance model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def classify_stance_batch(tokenizer, model, source_text: str,
                          replies: list[str]) -> list[str]:
    stances = []
    for reply in replies:
        user_msg = (
            f"Source claim: {source_text[:300]}\n\n"
            f"Reply: {reply[:200]}\n\n"
            "Stance:"
        )
        messages = [
            {"role": "system", "content": STANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()

        matched = "neutral"
        for s in STANCES:
            if s in generated:
                matched = s
                break
        stances.append(matched)
    return stances


def bucket_and_sample(replies_with_stance: list[dict],
                      max_per_bucket: int = MAX_PER_BUCKET) -> list[dict]:
    buckets: dict[str, list] = defaultdict(list)
    for r in replies_with_stance:
        buckets[r["stance"]].append(r)

    selected = []
    for stance in STANCES:
        bucket = buckets[stance]
        by_len = sorted(bucket, key=lambda r: len(r["text"]), reverse=True)
        selected.extend(by_len[:max_per_bucket])

    return selected[:MAX_TOTAL_REPLIES]


def compute_stance_dist(all_replies: list[dict]) -> dict:
    total = len(all_replies)
    if total == 0:
        return {s: 0.0 for s in STANCES}
    counts = defaultdict(int)
    for r in all_replies:
        counts[r["stance"]] += 1
    return {s: round(counts[s] / total, 3) for s in STANCES}


def format_basepack_v2(source_text: str, selected: list[dict],
                       stats: dict, stance_dist: dict) -> str:
    lines = ["[SOURCE]", source_text, ""]

    stance_counters: dict[str, int] = defaultdict(int)
    for item in selected:
        stance = item["stance"].upper()
        stance_counters[item["stance"]] += 1
        idx = stance_counters[item["stance"]]
        lines += [f"[REPLY-{stance}_{idx}]", item["text"], ""]

    dist_str = " | ".join(f"{s}={stance_dist.get(s, 0.0)}" for s in STANCES)
    lines += [
        "[STATS]",
        (
            f"reply_count={stats.get('num_replies', 0)} | "
            f"depth={stats.get('max_depth', 0)} | "
            f"branches={stats.get('num_branches', 0)}"
        ),
        f"[STANCE_DIST] {dist_str}",
    ]
    return "\n".join(lines)


def build_for_split(input_path: Path, output_path: Path,
                    tokenizer, model):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resume_ids: set = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    resume_ids.add(obj["event_id"])
        print(f"  Resuming: {len(resume_ids)} done.")

    events = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    to_process = [e for e in events if e["event_id"] not in resume_ids]
    print(f"  Processing {len(to_process)} events from {input_path}")

    with open(output_path, "a") as fout:
        for i, e in enumerate(to_process, 1):
            replies = e.get("replies", [])
            source_text = e.get("source_text", "")
            stats = e.get("meta", e.get("stats", {}))

            reply_texts = [r.get("text", "").strip() for r in replies if r.get("text", "").strip()]

            if reply_texts and model is not None:
                stances = classify_stance_batch(tokenizer, model, source_text, reply_texts)
                replies_with_stance = [
                    {"text": t, "stance": s} for t, s in zip(reply_texts, stances)
                ]
            else:
                replies_with_stance = [{"text": t, "stance": "neutral"} for t in reply_texts]

            stance_dist = compute_stance_dist(replies_with_stance)
            selected = bucket_and_sample(replies_with_stance)
            basepack_text = format_basepack_v2(source_text, selected, stats, stance_dist)

            record = {
                "event_id": e["event_id"],
                "topic": e.get("topic", ""),
                "label": e["label"],
                "basepack_text": basepack_text,
                "source_text": source_text,
                "selected_replies": selected,
                "stats": {
                    "num_replies": stats.get("num_replies", len(replies)),
                    "max_depth": stats.get("max_depth", 0),
                    "num_branches": stats.get("num_branches", 0),
                    "time_span": stats.get("time_span", "unknown"),
                },
                "stance_dist": stance_dist,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if i % 50 == 0:
                print(f"    Progress: {i}/{len(to_process)}")

    print(f"  -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build BasePack-v2-Stance representations")
    parser.add_argument("--input_dir", default="data/processed")
    parser.add_argument("--output_dir", default="basepack_v2")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--stance_model", default="Qwen/Qwen3-1.7B",
                        help="Model for zero-shot stance classification")
    parser.add_argument("--no_stance_model", action="store_true",
                        help="Skip stance model; label all replies as neutral")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = None, None
    if not args.no_stance_model:
        tokenizer, model = load_stance_model(args.stance_model)

    for split in args.splits:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            print(f"[WARN] Not found: {input_path}, skipping.")
            continue
        output_path = output_dir / f"basepack_{split}.jsonl"
        print(f"\nBuilding BasePack-v2 for {split}...")
        build_for_split(input_path, output_path, tokenizer, model)

    print("\nBasePack-v2 build complete.")


if __name__ == "__main__":
    main()
