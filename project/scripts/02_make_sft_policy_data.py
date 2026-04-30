"""
02_make_sft_policy_data.py
Build SFT training data for Prompt Policy from BasePack files.

Each event is mapped to one of 5 hint templates based on heuristic rules.
Output is LLaMA-Factory alpaca format.

Input:  basepack/basepack_{train,val}.jsonl
Output: policy_data/sft/{train,val}.json
"""

import json
import re
import argparse
from pathlib import Path

INSTRUCTION = (
    "Generate a concise focus_hint for LLM-assisted rumor analysis. "
    "Return only a JSON object with the key focus_hint."
)

HINT_TEMPLATES = {
    "balanced_focus": (
        "Summarize the central claim, compare supportive and skeptical replies, "
        "check source grounding, and note whether the evidence is concrete or mainly emotional."
    ),
    "query_focus": (
        "Prioritize questioning replies and uncertainty signals. Check what details are missing "
        "from the source claim, especially time, location, actor, source attribution, and evidence form."
    ),
    "deny_focus": (
        "Prioritize replies that deny, correct, or challenge the source claim. Compare them with "
        "supportive replies and identify whether the disagreement is based on evidence or speculation."
    ),
    "source_grounding_focus": (
        "Focus on whether the source post contains verifiable information such as named entities, "
        "time, location, source links, or attribution. Be cautious if the claim is vague."
    ),
    "conflict_propagation_focus": (
        "Focus on conflict and propagation signals. Check whether replies show agreement, denial, "
        "questions, repeated claims, or attention-driven reactions without concrete evidence."
    ),
}

DENY_KEYWORDS = re.compile(
    r"\b(fake|false|hoax|not true|debunked|wrong|incorrect|lie|liar|fabricated|misinformation)\b",
    re.IGNORECASE,
)

ENTITY_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,}|\d{4}[-/]\d{2}[-/]\d{2}|\d+:\d+)\b"
)


def select_hint_template(event: dict) -> str:
    replies = event.get("selected_replies", [])
    source_text = event.get("source_text", "")
    stats = event.get("stats", {})

    reply_text = " ".join(replies)
    question_count = reply_text.count("?")

    if DENY_KEYWORDS.search(reply_text):
        return "deny_focus"

    if question_count >= 3:
        return "query_focus"

    entities = ENTITY_PATTERN.findall(source_text)
    word_count = len(source_text.split())
    if word_count < 20 or len(entities) < 2:
        return "source_grounding_focus"

    num_replies = stats.get("num_replies", 0)
    max_depth = stats.get("max_depth", 0)
    num_branches = stats.get("num_branches", 0)
    if num_replies > 30 or max_depth > 4 or num_branches > 6:
        return "conflict_propagation_focus"

    return "balanced_focus"


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


def build_sft_sample(event: dict) -> dict:
    hint_type = select_hint_template(event)
    hint_text = HINT_TEMPLATES[hint_type]
    policy_input = build_policy_input(event)
    output = json.dumps({"focus_hint": hint_text}, ensure_ascii=False)
    return {
        "instruction": INSTRUCTION,
        "input": policy_input,
        "output": output,
    }


def process_split(input_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            samples.append(build_sft_sample(event))

    with open(output_path, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"  -> {output_path}  ({len(samples)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Build SFT policy data")
    parser.add_argument("--basepack_dir", default="basepack")
    parser.add_argument("--output_dir", default="policy_data/sft")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    basepack_dir = Path(args.basepack_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        input_path = basepack_dir / f"basepack_{split}.jsonl"
        if not input_path.exists():
            print(f"[WARN] Not found: {input_path}, skipping.")
            continue
        output_path = output_dir / f"{split}.json"
        print(f"Building SFT data for {split}...")
        process_split(input_path, output_path)

    print("SFT data build complete.")


if __name__ == "__main__":
    main()
