"""
05_make_dpo_pairs.py
Construct DPO chosen/rejected pairs from scored candidate hints.

Reward formula:
  correct       = 1 if weak_label == gold_label else 0
  signed_conf   = weak_confidence if correct else -weak_confidence
  json_bonus    = 1 if llm_json_valid else 0
  cost_penalty  = log(1 + output_tokens / 100)
  reward = 2.0 * correct + 1.0 * signed_conf + 0.3 * json_bonus - 0.1 * cost_penalty

Pair rule:
  chosen   = candidate with highest reward
  rejected = candidate with lowest reward
  kept only if reward(chosen) - reward(rejected) >= 0.5
            AND chosen must have valid JSON

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

LABEL_MAP = {"true": 0, "fake": 1}

HINT_TEMPLATES = {
    "h0_heuristic": (
        "Focus on the central claim, supporting and refuting replies, conflict among replies, "
        "source grounding, and whether the discussion provides concrete evidence or only emotional reactions."
    ),
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


def compute_reward(record: dict) -> float:
    gold = int(record["gold_label"])
    json_valid = int(record.get("llm_json_valid", 0))
    out_tok = int(record.get("output_tokens", 100))

    if not json_valid or record.get("weak_label") is None:
        return -2.0

    weak_int = LABEL_MAP.get(str(record["weak_label"]).lower(), -1)
    if weak_int == -1:
        return -2.0

    conf = float(record.get("weak_confidence", 0.5))
    correct = 1 if weak_int == gold else 0
    signed_conf = conf if correct else -conf
    cost_penalty = math.log(1 + out_tok / 100)

    return 2.0 * correct + 1.0 * signed_conf + 0.3 * json_valid - 0.1 * cost_penalty



def make_dpo_pairs(scored_path: Path, basepack_path: Path | None,
                   output_path: Path, min_margin: float = 0.5):
    basepack_lookup: dict[str, str] = {}
    if basepack_path and basepack_path.exists():
        with open(basepack_path) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    basepack_lookup[e["event_id"]] = e.get("basepack_text", "")

    by_event: dict[str, list] = defaultdict(list)
    with open(scored_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                by_event[rec["event_id"]].append(rec)

    pairs = []
    skipped_no_margin = 0
    skipped_invalid_chosen = 0

    for event_id, records in by_event.items():
        for rec in records:
            rec["_reward"] = compute_reward(rec)

        valid = [r for r in records if r["_reward"] > -2.0]
        if len(valid) < 2:
            continue

        sorted_recs = sorted(valid, key=lambda r: r["_reward"], reverse=True)
        chosen_rec = sorted_recs[0]
        rejected_rec = sorted_recs[-1]

        if chosen_rec["_reward"] - rejected_rec["_reward"] < min_margin:
            skipped_no_margin += 1
            continue

        if not chosen_rec.get("llm_json_valid", 0):
            skipped_invalid_chosen += 1
            continue

        chosen_hint = chosen_rec.get("focus_hint") or HINT_TEMPLATES.get(chosen_rec["hint_id"], "")
        rejected_hint = rejected_rec.get("focus_hint") or HINT_TEMPLATES.get(rejected_rec["hint_id"], "")

        if not chosen_hint or not rejected_hint:
            continue

        policy_input = basepack_lookup.get(event_id, "")
        if not policy_input:
            policy_input = f"[Event: {event_id}]"

        pairs.append({
            "instruction": INSTRUCTION,
            "input": policy_input,
            "chosen": json.dumps({"focus_hint": chosen_hint}, ensure_ascii=False),
            "rejected": json.dumps({"focus_hint": rejected_hint}, ensure_ascii=False),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"DPO pairs: {len(pairs)} written to {output_path}")
    print(f"  Skipped (no margin): {skipped_no_margin}")
    print(f"  Skipped (invalid chosen): {skipped_invalid_chosen}")


def main():
    parser = argparse.ArgumentParser(description="Build DPO pairs from scored candidates")
    parser.add_argument("--scored", default="policy_data/dpo/scored_candidates_train.jsonl")
    parser.add_argument("--basepack", default="basepack/basepack_train.jsonl",
                        help="Used to reconstruct policy input text")
    parser.add_argument("--output", default="policy_data/dpo/train.json")
    parser.add_argument("--min_margin", type=float, default=0.5)
    args = parser.parse_args()

    basepack_path = Path(args.basepack) if args.basepack else None
    make_dpo_pairs(Path(args.scored), basepack_path, Path(args.output), args.min_margin)


if __name__ == "__main__":
    main()
