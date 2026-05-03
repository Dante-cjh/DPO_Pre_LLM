"""
08_build_slm_dataset.py
Build SLM input files by concatenating BasePack + LLM_AUG.

Methods:
  basepack_only  - only BasePack, no LLM_AUG
  heuristic_pre  - BasePack + LLM_AUG from heuristic hint
  sft_hint_pre   - BasePack + LLM_AUG from SFT policy hint
  dpo_hint_pre   - BasePack + LLM_AUG from DPO policy hint

If LLM_AUG is null (parse failure), falls back to BasePack only.

Output per line:
{
  "event_id": str,
  "text": str,      # input for DeBERTa
  "label": int
}
"""

import json
import argparse
from pathlib import Path

LLM_AUG_MAP = {
    "heuristic_pre": "llm_aug/heuristic/{split}.jsonl",
    "sft_hint_pre": "llm_aug/sft_hint/{split}.jsonl",
    "dpo_hint_pre": "llm_aug/dpo_hint/{split}.jsonl",
}


def format_llm_aug(aug: dict) -> str:
    lines = ["[LLM_AUG]"]

    claim = aug.get("claim_summary", "")
    if claim:
        lines.append(f"Claim: {claim}")

    sup = aug.get("supporting_signals", [])
    if sup:
        lines.append(f"Supporting: {' | '.join(str(s) for s in sup)}")

    ref = aug.get("refuting_signals", [])
    if ref:
        lines.append(f"Refuting: {' | '.join(str(s) for s in ref)}")

    conf = aug.get("conflict_summary", "")
    if conf:
        lines.append(f"Conflict: {conf}")

    vf = aug.get("verification_focus", [])
    if vf:
        lines.append(f"VerificationFocus: {' | '.join(str(v) for v in vf)}")

    unc = aug.get("uncertainty_note", "")
    if unc:
        lines.append(f"Uncertainty: {unc}")

    return "\n".join(lines)


def build_split(basepack_path: Path, aug_path: Path | None,
                output_path: Path, method: str):
    aug_lookup: dict[str, dict | None] = {}
    if aug_path is not None and aug_path.exists():
        with open(aug_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    aug_lookup[obj["event_id"]] = obj.get("llm_aug")
    elif aug_path is not None:
        print(f"[WARN] LLM_AUG file not found: {aug_path}. Falling back to BasePack only.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fallback_count = 0

    with open(basepack_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            event = json.loads(line)
            eid = event["event_id"]
            basepack_text = event["basepack_text"]

            if method == "basepack_only" or aug_path is None:
                text = f"[SOURCE_AND_REPLIES]\n{basepack_text}"
            else:
                aug = aug_lookup.get(eid)
                if aug is not None:
                    aug_text = format_llm_aug(aug)
                    text = f"[SOURCE_AND_REPLIES]\n{basepack_text}\n\n{aug_text}"
                else:
                    fallback_count += 1
                    text = f"[SOURCE_AND_REPLIES]\n{basepack_text}"

            record = {
                "event_id": eid,
                "text": text,
                "label": event["label"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"  -> {output_path}  ({count} samples, {fallback_count} LLM_AUG fallbacks)")


def main():
    parser = argparse.ArgumentParser(description="Build SLM dataset")
    parser.add_argument("--method",
                        choices=["basepack_only", "heuristic_pre", "sft_hint_pre", "dpo_hint_pre"],
                        default=None,
                        help="If set, processes all splits for this method")
    parser.add_argument("--basepack", default=None,
                        help="Single basepack file (manual mode)")
    parser.add_argument("--llm_aug", default=None,
                        help="Single LLM_AUG file (manual mode); omit for basepack_only")
    parser.add_argument("--output", default=None,
                        help="Output file (manual mode)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    if args.method:
        aug_template = LLM_AUG_MAP.get(args.method)
        for split in args.splits:
            basepack_path = Path(f"basepack_v2/basepack_{split}.jsonl")
            if not basepack_path.exists():
                print(f"[WARN] Not found: {basepack_path}, skipping.")
                continue
            aug_path = Path(aug_template.replace("{split}", split)) if aug_template else None
            output_path = Path(f"slm_data/{args.method}/{split}.jsonl")
            print(f"Building SLM data: method={args.method}, split={split} (basepack_v2)")
            build_split(basepack_path, aug_path, output_path, args.method)
    else:
        if not args.basepack or not args.output:
            raise ValueError("Provide --method OR --basepack + --output")
        aug_path = Path(args.llm_aug) if args.llm_aug else None
        method = "basepack_only" if aug_path is None else "custom"
        build_split(Path(args.basepack), aug_path, Path(args.output), method)

    print("SLM dataset build complete.")


if __name__ == "__main__":
    main()
