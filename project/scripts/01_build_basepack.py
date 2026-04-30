"""
01_build_basepack.py
Build BasePack text representations from raw split files.

Expects data/processed/{train,val,test}.jsonl produced by the FN_Cascaded pipeline
(binary_events split into train/val/test via 02_make_splits.py).

Output: basepack/basepack_{split}.jsonl
Each line:
{
  "event_id": str,
  "topic": str,
  "label": int,        # 0=True, 1=Fake
  "basepack_text": str,
  "source_text": str,
  "selected_replies": [str],
  "stats": {"num_replies": int, "max_depth": int, "num_branches": int, "time_span": str}
}
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def build_branch_map(structure: dict) -> dict:
    branch_map: dict[str, str] = {}

    def _walk(node, branch_root):
        if not isinstance(node, dict):
            return
        for tid, children in node.items():
            root = branch_root if branch_root else tid
            branch_map[tid] = root
            _walk(children, root)

    _walk(structure, None)
    return branch_map


def select_replies(replies: list, structure: dict, max_replies: int = 8) -> list:
    if not replies:
        return []

    timed = sorted(replies, key=lambda r: r.get("time", ""))
    selected_ids: set = set()
    selected_texts: list = []

    def add(r):
        tid = r.get("tweet_id", "")
        text = r.get("text", "").strip()
        if tid not in selected_ids and text:
            selected_ids.add(tid)
            selected_texts.append(text)

    for r in timed[:3]:
        add(r)

    by_len = sorted(replies, key=lambda r: len(r.get("text", "")), reverse=True)
    count = 0
    for r in by_len:
        if r.get("tweet_id", "") not in selected_ids:
            add(r)
            count += 1
        if count >= 3:
            break

    branch_map = build_branch_map(structure)
    seen_branches: set = set()
    for r in timed:
        if len(seen_branches) >= 2:
            break
        br = branch_map.get(r.get("tweet_id", ""), r.get("tweet_id", ""))
        if br not in seen_branches and r.get("tweet_id", "") not in selected_ids:
            seen_branches.add(br)
            add(r)

    return selected_texts[:max_replies]


def format_basepack(source_text: str, selected_replies: list, stats: dict) -> str:
    lines = ["[SOURCE]", source_text, ""]
    for i, reply in enumerate(selected_replies, 1):
        lines += [f"[REPLY_{i}]", reply, ""]
    lines += [
        "[STATS]",
        f"reply_count={stats.get('num_replies', 0)}",
        f"max_depth={stats.get('max_depth', 0)}",
        f"num_branches={stats.get('num_branches', 0)}",
        f"time_span={stats.get('time_span', 'unknown')}",
    ]
    return "\n".join(lines)


def build_for_split(input_path: Path, output_path: Path, max_replies: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            e = json.loads(line)
            selected = select_replies(
                e.get("replies", []),
                e.get("structure", {}),
                max_replies,
            )
            stats = e.get("meta", e.get("stats", {}))
            basepack_text = format_basepack(e["source_text"], selected, stats)
            record = {
                "event_id": e["event_id"],
                "topic": e.get("topic", ""),
                "label": e["label"],
                "basepack_text": basepack_text,
                "source_text": e["source_text"],
                "selected_replies": selected,
                "stats": {
                    "num_replies": stats.get("num_replies", 0),
                    "max_depth": stats.get("max_depth", 0),
                    "num_branches": stats.get("num_branches", 0),
                    "time_span": stats.get("time_span", "unknown"),
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"  -> {output_path}  ({count} events)")


def main():
    parser = argparse.ArgumentParser(description="Build BasePack representations")
    parser.add_argument("--input_dir", default="data/processed",
                        help="Directory with {train,val,test}.jsonl splits")
    parser.add_argument("--output_dir", default="basepack")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max_replies", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            print(f"[WARN] Not found: {input_path}, skipping.")
            continue
        output_path = output_dir / f"basepack_{split}.jsonl"
        print(f"Building BasePack for {split}...")
        build_for_split(input_path, output_path, args.max_replies)

    print("BasePack build complete.")


if __name__ == "__main__":
    main()
