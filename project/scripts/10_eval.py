"""
10_eval.py
Aggregate evaluation results from all four SLM experiments.

Reads:
  slm_outputs/{method}/metrics.json
  llm_aug/{method}/{split}.jsonl   (for JSON parse rate and token stats)

Writes:
  outputs/metrics/main_results.json
  outputs/metrics/main_results.md
"""

import json
import argparse
from pathlib import Path

METHODS = [
    ("basepack_only", "BasePack only", None),
    ("heuristic_pre", "Heuristic Pre", "heuristic"),
    ("sft_hint_pre",  "SFT Hint Pre",  "sft_hint"),
    ("dpo_hint_pre",  "DPO Hint Pre",  "dpo_hint"),
]


def load_metrics(method: str) -> dict | None:
    path = Path(f"slm_outputs/{method}/metrics.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_llm_aug_stats(llm_method: str | None, splits=("train", "val", "test")) -> dict:
    if llm_method is None:
        return {}

    total = 0
    valid = 0
    total_in = 0
    total_out = 0

    for split in splits:
        path = Path(f"llm_aug/{llm_method}/{split}.jsonl")
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                total += 1
                if rec.get("json_valid", 0):
                    valid += 1

    parse_rate = round(valid / total, 4) if total > 0 else 0.0

    test_path = Path(f"llm_aug/{llm_method}/test.jsonl")
    n_test = 0
    if test_path.exists():
        with open(test_path) as f:
            for line in f:
                if line.strip():
                    n_test += 1

    return {
        "json_parse_rate": parse_rate,
        "total_aug_calls": total,
        "valid_aug_calls": valid,
    }


def fmt(v, decimals=4) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def build_results():
    results = {}
    for method, label, llm_method in METHODS:
        m = load_metrics(method)
        stats = compute_llm_aug_stats(llm_method)
        results[method] = {
            "label": label,
            "metrics": m,
            "aug_stats": stats,
        }
    return results


def compute_gains(results: dict) -> list:
    def get(method, key):
        m = results.get(method, {}).get("metrics")
        return m.get(key) if m else None

    gains = [
        {
            "comparison": "Heuristic - BasePack",
            "delta_acc":    _delta(get("heuristic_pre", "accuracy"),   get("basepack_only", "accuracy")),
            "delta_macro":  _delta(get("heuristic_pre", "macro_f1"),   get("basepack_only", "macro_f1")),
            "delta_f1fake": _delta(get("heuristic_pre", "f1_fake"),    get("basepack_only", "f1_fake")),
        },
        {
            "comparison": "SFT Hint - Heuristic",
            "delta_acc":    _delta(get("sft_hint_pre", "accuracy"),    get("heuristic_pre", "accuracy")),
            "delta_macro":  _delta(get("sft_hint_pre", "macro_f1"),    get("heuristic_pre", "macro_f1")),
            "delta_f1fake": _delta(get("sft_hint_pre", "f1_fake"),     get("heuristic_pre", "f1_fake")),
        },
        {
            "comparison": "DPO Hint - SFT Hint",
            "delta_acc":    _delta(get("dpo_hint_pre", "accuracy"),    get("sft_hint_pre", "accuracy")),
            "delta_macro":  _delta(get("dpo_hint_pre", "macro_f1"),    get("sft_hint_pre", "macro_f1")),
            "delta_f1fake": _delta(get("dpo_hint_pre", "f1_fake"),     get("sft_hint_pre", "f1_fake")),
        },
        {
            "comparison": "DPO Hint - Heuristic",
            "delta_acc":    _delta(get("dpo_hint_pre", "accuracy"),    get("heuristic_pre", "accuracy")),
            "delta_macro":  _delta(get("dpo_hint_pre", "macro_f1"),    get("heuristic_pre", "macro_f1")),
            "delta_f1fake": _delta(get("dpo_hint_pre", "f1_fake"),     get("heuristic_pre", "f1_fake")),
        },
    ]
    return gains


def _delta(a, b):
    if a is None or b is None:
        return None
    return round((a - b) * 100, 2)


def _fmt_delta(v) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}"


def write_markdown(results: dict, gains: list, out_path: Path):
    lines = [
        "# Main Results",
        "",
        "| Method | Accuracy | Macro-F1 | F1-Fake | F1-True | JSON Parse Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, label, _ in METHODS:
        r = results[method]
        m = r.get("metrics") or {}
        aug = r.get("aug_stats") or {}
        parse_str = fmt(aug.get("json_parse_rate")) if aug else "—"
        lines.append(
            f"| {label} | {fmt(m.get('accuracy'))} | {fmt(m.get('macro_f1'))} | "
            f"{fmt(m.get('f1_fake'))} | {fmt(m.get('f1_true'))} | {parse_str} |"
        )

    lines += [
        "",
        "# Key Comparisons (pp = percentage points)",
        "",
        "| Comparison | ΔAcc (pp) | ΔMacro-F1 (pp) | ΔF1-Fake (pp) |",
        "|---|---:|---:|---:|",
    ]
    for g in gains:
        lines.append(
            f"| {g['comparison']} | {_fmt_delta(g['delta_acc'])} | "
            f"{_fmt_delta(g['delta_macro'])} | {_fmt_delta(g['delta_f1fake'])} |"
        )

    lines += [
        "",
        "# Notes",
        "",
        "- DPO > SFT > Heuristic: preference training effective.",
        "- SFT > Heuristic but DPO flat: check DPO reward design.",
        "- Heuristic best: check SFT template quality and scoring reward.",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument("--output_dir", default="outputs/metrics")
    args = parser.parse_args()

    results = build_results()
    gains = compute_gains(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_out = output_dir / "main_results.json"
    with open(json_out, "w") as f:
        json.dump({"results": results, "gains": gains}, f, indent=2, ensure_ascii=False)
    print(f"JSON:     {json_out}")

    write_markdown(results, gains, output_dir / "main_results.md")

    print("\n=== Summary ===")
    for method, label, _ in METHODS:
        m = (results[method].get("metrics") or {})
        acc = fmt(m.get("accuracy"))
        mf1 = fmt(m.get("macro_f1"))
        print(f"  {label:20s}  Acc={acc}  Macro-F1={mf1}")


if __name__ == "__main__":
    main()
