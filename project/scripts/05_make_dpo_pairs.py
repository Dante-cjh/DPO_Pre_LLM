"""
05_make_dpo_pairs.py
Construct DPO chosen/rejected pairs from scored candidate hints.

Reward values are pre-computed by 04_score_candidate_hints.py (SLM-aligned reward).
This script applies two stabilization steps before pairing:

  1. Within-event normalization:
       For each event, subtract the mean reward and divide by std across all candidates.
       This removes the bias from events that are inherently harder (all rewards low).

  2. Pair selection:
       chosen   = candidate with highest normalized reward
       rejected = candidate with lowest normalized reward
       Kept only if raw margin (before normalization) >= min_margin AND
       chosen must have llm_json_valid=1.

Input:  policy_data/dpo/scored_candidates_train.jsonl
Output: policy_data/dpo/train.json  (LLaMA-Factory ranking alpaca format)
"""

import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

INSTRUCTION = (
    "Generate a concise focus_hint for LLM-assisted rumor analysis. "
    "Return only a JSON object with the key focus_hint."
)


def normalize_within_event(records: list[dict]) -> list[dict]:
    rewards = [r["reward"] for r in records]
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(variance) if variance > 0 else 1.0
    for rec in records:
        rec["_norm_reward"] = (rec["reward"] - mean) / std
    return records


def build_policy_input_from_event(record: dict) -> str:
    source_text = record.get("source_text", "")
    replies = record.get("selected_replies", [])
    stats = record.get("stats", {})
    stance_dist = record.get("stance_dist", {})

    lines = ["SOURCE:", source_text, "", "REPLIES:"]
    for i, r in enumerate(replies, 1):
        text = r["text"] if isinstance(r, dict) else r
        lines.append(f"{i}. {text[:150]}")

    lines.append("")
    lines.append(
        f"STATS:\nreply_count={stats.get('num_replies', 0)} | "
        f"depth={stats.get('max_depth', 0)} | "
        f"branches={stats.get('num_branches', 0)}"
    )
    if stance_dist:
        dist_str = " | ".join(
            f"{s}={stance_dist.get(s, 0.0)}"
            for s in ["support", "deny", "query", "neutral"]
        )
        lines.append(f"STANCE_DIST: {dist_str}")
    return "\n".join(lines)


def make_dpo_pairs(scored_path: Path, basepack_path: Path | None,
                   output_path: Path, min_margin: float = 1.0):
    basepack_lookup: dict[str, dict] = {}
    if basepack_path and basepack_path.exists():
        with open(basepack_path) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    basepack_lookup[e["event_id"]] = e

    by_event: dict[str, list] = defaultdict(list)
    with open(scored_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                by_event[rec["event_id"]].append(rec)

    pairs = []
    skipped_no_margin = 0
    skipped_invalid_chosen = 0
    skipped_too_few = 0

    for event_id, records in by_event.items():
        valid = [r for r in records if "reward" in r]
        if len(valid) < 2:
            skipped_too_few += 1
            continue

        valid = normalize_within_event(valid)
        sorted_recs = sorted(valid, key=lambda r: r["_norm_reward"], reverse=True)
        chosen_rec = sorted_recs[0]
        rejected_rec = sorted_recs[-1]

        raw_margin = chosen_rec["reward"] - rejected_rec["reward"]
        if raw_margin < min_margin:
            skipped_no_margin += 1
            continue

        if not chosen_rec.get("llm_json_valid", 0):
            skipped_invalid_chosen += 1
            continue

        chosen_hint = chosen_rec.get("focus_hint", "")
        rejected_hint = rejected_rec.get("focus_hint", "")
        if not chosen_hint or not rejected_hint:
            continue

        bp_event = basepack_lookup.get(event_id, {})
        if bp_event:
            policy_input = build_policy_input_from_event(bp_event)
        else:
            policy_input = f"[Event: {event_id}]"

        pairs.append({
            "instruction": INSTRUCTION,
            "input": policy_input,
            "chosen": json.dumps({"focus_hint": chosen_hint}, ensure_ascii=False),
            "rejected": json.dumps({"focus_hint": rejected_hint}, ensure_ascii=False),
            "_meta": {
                "event_id": event_id,
                "chosen_hint_id": chosen_rec.get("hint_id"),
                "rejected_hint_id": rejected_rec.get("hint_id"),
                "chosen_reward": chosen_rec["reward"],
                "rejected_reward": rejected_rec["reward"],
                "raw_margin": round(raw_margin, 4),
                "chosen_slm_gain": chosen_rec.get("slm_gain", 0.0),
                "rejected_slm_gain": rejected_rec.get("slm_gain", 0.0),
            },
        })

    meta_path = output_path.with_suffix(".meta.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs_for_training = [
        {k: v for k, v in p.items() if k != "_meta"} for p in pairs
    ]
    with open(output_path, "w") as f:
        json.dump(pairs_for_training, f, ensure_ascii=False, indent=2)

    with open(meta_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p["_meta"], ensure_ascii=False) + "\n")

    print(f"DPO pairs: {len(pairs)} written to {output_path}")
    print(f"  Skipped (raw margin < {min_margin}): {skipped_no_margin}")
    print(f"  Skipped (invalid chosen): {skipped_invalid_chosen}")
    print(f"  Skipped (too few valid): {skipped_too_few}")
    print(f"  Pair metadata: {meta_path}")

    if pairs:
        avg_margin = sum(p["_meta"]["raw_margin"] for p in pairs) / len(pairs)
        avg_slm_gain_chosen = sum(p["_meta"]["chosen_slm_gain"] for p in pairs) / len(pairs)
        avg_slm_gain_rejected = sum(p["_meta"]["rejected_slm_gain"] for p in pairs) / len(pairs)
        print(f"  Avg raw margin:          {avg_margin:.4f}")
        print(f"  Avg slm_gain (chosen):   {avg_slm_gain_chosen:.4f}")
        print(f"  Avg slm_gain (rejected): {avg_slm_gain_rejected:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Build DPO pairs from scored candidates")
    parser.add_argument("--scored", default="policy_data/dpo/scored_candidates_train.jsonl")
    parser.add_argument("--basepack", default="basepack_v2/basepack_train.jsonl",
                        help="BasePack-v2 file to reconstruct policy input text")
    parser.add_argument("--output", default="policy_data/dpo/train.json")
    parser.add_argument("--min_margin", type=float, default=1.0,
                        help="Minimum raw reward margin between chosen and rejected")
    args = parser.parse_args()

    basepack_path = Path(args.basepack) if args.basepack else None
    make_dpo_pairs(Path(args.scored), basepack_path, Path(args.output), args.min_margin)


if __name__ == "__main__":
    main()
