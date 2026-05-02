"""
04_score_candidate_hints.py
Score each candidate hint using SLM-aligned reward.

Pipeline per event (12 candidates processed together):
  1. [LLM]  Call Qwen3-8B (local or API) to produce LLM_AUG for all 12 candidates
             API mode: concurrent calls via ThreadPoolExecutor
             Local mode: batched generation (all 12 candidates at once)
  2. [SLM]  Forward BasePack through frozen G0 SLM probe -> p_base[gold]  (cached once per event)
  3. [SLM]  Batch forward all 12 (BasePack + LLM_AUG) texts -> p_aug[gold] (single batched call)
  4. Compute slm_gain = log(p_aug + eps) - log(p_base + eps)
  5. Compute full reward:
       1.5 * slm_gain
     + 0.5 * acc_llm
     + 0.3 * format_ok
     + 0.2 * field_coverage
     - 0.1 * log(1 + out_tok/100)

Backends:
  --backend api    : OpenAI-compatible API with concurrent workers (DEFAULT, recommended)
  --backend local  : local Qwen3-8B HuggingFace inference (slow, for offline use)

Input:  policy_data/dpo/candidate_hints_{split}.jsonl
        slm_outputs/basepack_only/best_model/  (frozen G0 SLM probe)
Output: policy_data/dpo/scored_candidates_{split}.jsonl
"""

import json
import math
import re
import os
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

REQUIRED_FIELDS = [
    "claim_summary", "supporting_signals", "refuting_signals",
    "conflict_summary", "verification_focus", "uncertainty_note",
]

ADVISOR_SYSTEM_PROMPT = (
    "You are a misinformation analysis assistant. Analyze the given social media event and "
    "follow the provided focus hint. Return a JSON object with exactly these fields:\n"
    "claim_summary, supporting_signals, refuting_signals, conflict_summary, "
    "verification_focus, uncertainty_note, weak_label (\"true\" or \"fake\"), weak_confidence (0.5-1.0).\n"
    "Only use information present in the event. Do not invent facts."
)

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
SLM_MAX_LENGTH = 512
SLM_BATCH_SIZE = 32   # DeBERTa batch size for probe forward passes


# ── LLM_AUG generation ────────────────────────────────────────────────────────

def _build_user_msg(focus_hint: str, basepack_text: str) -> str:
    return (
        f"[FOCUS_HINT]\n{focus_hint}\n\n"
        f"[EVENT]\n{basepack_text}"
    )


def _parse_aug_response(raw: str) -> tuple[dict | None, int]:
    """Returns (parsed_dict_or_None, approx_output_token_count)."""
    out_tokens = len(raw.split()) if raw else 0
    if not raw:
        return None, 0
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
        return None, out_tokens
    if not all(f in obj for f in REQUIRED_FIELDS):
        return None, out_tokens
    if str(obj.get("weak_label", "")).lower() not in ("true", "fake"):
        return None, out_tokens
    try:
        float(obj["weak_confidence"])
    except (ValueError, TypeError):
        return None, out_tokens
    return obj, out_tokens


# ── API backend ───────────────────────────────────────────────────────────────

def _api_call_one(client: "OpenAI", model: str,
                  focus_hint: str, basepack_text: str,
                  max_retries: int, retry_delay: int) -> tuple[dict | None, int]:
    user_msg = _build_user_msg(focus_hint, basepack_text)
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
            out_tok = resp.usage.completion_tokens if resp.usage else len(raw.split())
            parsed, _ = _parse_aug_response(raw)
            return parsed, out_tok
        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                print(f"    [API error after {max_retries} retries] {e}")
    return None, 0


def generate_aug_batch_api(client: "OpenAI", model: str,
                           candidates: list[dict], basepack_text: str,
                           max_workers: int, max_retries: int,
                           retry_delay: int) -> list[tuple[dict | None, int]]:
    """Concurrently call API for all candidates of one event."""
    results: list[tuple[dict | None, int]] = [None] * len(candidates)

    def _call(idx: int, hint: str):
        return idx, _api_call_one(client, model, hint, basepack_text,
                                  max_retries, retry_delay)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(candidates))) as ex:
        futures = {
            ex.submit(_call, i, c["focus_hint"]): i
            for i, c in enumerate(candidates)
        }
        for fut in as_completed(futures):
            idx, result = fut.result()
            results[idx] = result

    return results


# ── Local backend ─────────────────────────────────────────────────────────────

def load_llm_local(model_name: str):
    from transformers import AutoModelForCausalLM
    print(f"Loading local Qwen3-8B: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def generate_aug_batch_local(llm_tokenizer, llm_model,
                              candidates: list[dict],
                              basepack_text: str,
                              max_new_tokens: int = 300) -> list[tuple[dict | None, int]]:
    """Batched local generation for all candidates of one event."""
    texts = []
    for cand in candidates:
        user_msg = _build_user_msg(cand["focus_hint"], basepack_text)
        messages = [
            {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        texts.append(llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        ))

    # Left-pad for batch decode
    llm_tokenizer.padding_side = "left"
    inputs = llm_tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=2048).to(llm_model.device)
    with torch.no_grad():
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=llm_tokenizer.pad_token_id,
        )

    results = []
    in_len = inputs["input_ids"].shape[1]
    for i in range(len(candidates)):
        raw = llm_tokenizer.decode(
            output_ids[i][in_len:], skip_special_tokens=True
        ).strip()
        parsed, out_tok = _parse_aug_response(raw)
        results.append((parsed, out_tok))
    return results


# ── SLM probe (batched, p_base cached per event) ──────────────────────────────

def load_slm_probe(model_path: str):
    print(f"Loading frozen SLM probe: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, torch_dtype=torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model, device


def _slm_log_probs_batch(slm_tokenizer, slm_model, device: str,
                          texts: list[str], gold_label: int) -> list[float]:
    """Forward a batch of texts through SLM probe, return log P(gold) for each."""
    enc = slm_tokenizer(
        texts, truncation=True, max_length=SLM_MAX_LENGTH,
        padding=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = slm_model(**enc).logits          # (B, 2)
    log_probs = F.log_softmax(logits, dim=-1)     # (B, 2)
    return log_probs[:, gold_label].cpu().tolist()


def compute_slm_gains_for_event(slm_tokenizer, slm_model, device: str,
                                 basepack_text: str,
                                 aug_texts: list[str | None],
                                 gold_label: int) -> list[float]:
    """
    Compute slm_gain for all candidates of one event.
    p_base is computed once and cached; aug texts are batched.
    """
    base_input = f"[SOURCE_AND_REPLIES]\n{basepack_text}"
    [log_p_base] = _slm_log_probs_batch(
        slm_tokenizer, slm_model, device, [base_input], gold_label
    )

    gains = []
    valid_texts = []
    valid_indices = []
    for i, aug_text in enumerate(aug_texts):
        if aug_text is not None:
            valid_texts.append(f"[SOURCE_AND_REPLIES]\n{basepack_text}\n\n{aug_text}")
            valid_indices.append(i)

    # Batch forward all valid aug texts
    valid_log_probs: list[float] = []
    for batch_start in range(0, len(valid_texts), SLM_BATCH_SIZE):
        batch = valid_texts[batch_start:batch_start + SLM_BATCH_SIZE]
        valid_log_probs.extend(
            _slm_log_probs_batch(slm_tokenizer, slm_model, device, batch, gold_label)
        )

    lp_map = {idx: lp for idx, lp in zip(valid_indices, valid_log_probs)}
    for i in range(len(aug_texts)):
        if i in lp_map:
            gains.append(lp_map[i] - log_p_base)
        else:
            gains.append(0.0)
    return gains


# ── Reward ────────────────────────────────────────────────────────────────────

def format_llm_aug(aug: dict) -> str:
    lines = ["[LLM_AUG]"]
    if aug.get("claim_summary"):
        lines.append(f"Claim: {aug['claim_summary']}")
    sup = aug.get("supporting_signals", [])
    if sup:
        lines.append(f"Supporting: {' | '.join(str(s) for s in sup)}")
    ref = aug.get("refuting_signals", [])
    if ref:
        lines.append(f"Refuting: {' | '.join(str(s) for s in ref)}")
    if aug.get("conflict_summary"):
        lines.append(f"Conflict: {aug['conflict_summary']}")
    vf = aug.get("verification_focus", [])
    if vf:
        lines.append(f"VerificationFocus: {' | '.join(str(v) for v in vf)}")
    if aug.get("uncertainty_note"):
        lines.append(f"Uncertainty: {aug['uncertainty_note']}")
    return "\n".join(lines)


def compute_reward(slm_gain: float, aug: dict | None,
                   gold_label: int, out_tokens: int) -> float:
    format_ok = 1.0 if aug is not None else 0.0
    field_coverage = acc_llm = 0.0
    if aug is not None:
        non_empty = sum(1 for f in REQUIRED_FIELDS if aug.get(f) and str(aug[f]).strip())
        field_coverage = non_empty / len(REQUIRED_FIELDS)
        weak_label = str(aug.get("weak_label", "")).lower()
        pred = {"true": 0, "fake": 1}.get(weak_label, -1)
        acc_llm = 1.0 if pred == gold_label else 0.0
    cost_penalty = math.log(1 + out_tokens / 100)
    return (
        1.5 * slm_gain
        + 0.5 * acc_llm
        + 0.3 * format_ok
        + 0.2 * field_coverage
        - 0.1 * cost_penalty
    )


# ── Main scoring loop ─────────────────────────────────────────────────────────

def score_split(candidate_path: Path, output_path: Path,
                backend: str, llm_model_name: str, slm_probe_path: str,
                api_client: "OpenAI | None", api_model: str,
                api_workers: int, api_max_retries: int, api_retry_delay: int):

    # Load models
    llm_tokenizer = llm_model = None
    if backend == "local":
        llm_tokenizer, llm_model = load_llm_local(llm_model_name)

    slm_tokenizer, slm_model, slm_device = load_slm_probe(slm_probe_path)

    # Load events
    events = []
    with open(candidate_path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    resume_keys: set = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    resume_keys.add((obj["event_id"], obj["hint_id"]))
        print(f"Resuming: {len(resume_keys)} records already scored.")

    total_candidates = sum(len(e["candidates"]) for e in events)
    done = len(resume_keys)
    t0 = time.time()

    with open(output_path, "a") as fout:
        for event_idx, event in enumerate(events):
            eid = event["event_id"]
            gold_label = int(event["label"])
            basepack_text = event["basepack_text"]
            candidates = event["candidates"]

            # Skip events where all candidates are already scored
            pending = [c for c in candidates if (eid, c["hint_id"]) not in resume_keys]
            if not pending:
                continue

            # ── Step 1: Generate LLM_AUG for all pending candidates ──────────
            if backend == "api":
                aug_results = generate_aug_batch_api(
                    api_client, api_model, pending, basepack_text,
                    api_workers, api_max_retries, api_retry_delay,
                )
            else:
                aug_results = generate_aug_batch_local(
                    llm_tokenizer, llm_model, pending, basepack_text,
                )

            # ── Step 2: Batch SLM probe (p_base cached, all aug texts in one pass) ──
            aug_texts = [
                format_llm_aug(aug) if aug is not None else None
                for aug, _ in aug_results
            ]
            slm_gains = compute_slm_gains_for_event(
                slm_tokenizer, slm_model, slm_device,
                basepack_text, aug_texts, gold_label,
            )

            # ── Step 3: Write records ─────────────────────────────────────────
            for i, (cand, (aug, out_tokens), slm_gain) in enumerate(
                zip(pending, aug_results, slm_gains)
            ):
                reward = compute_reward(slm_gain, aug, gold_label, out_tokens)
                record = {
                    "event_id": eid,
                    "hint_id": cand["hint_id"],
                    "focus_hint": cand["focus_hint"],
                    "gold_label": gold_label,
                    "llm_json_valid": 1 if aug is not None else 0,
                    "weak_label": str(aug.get("weak_label", "")).lower() if aug else None,
                    "weak_confidence": float(aug.get("weak_confidence", 0.5)) if aug else None,
                    "slm_gain": round(slm_gain, 6),
                    "reward": round(reward, 6),
                    "output_tokens": out_tokens,
                    "analysis": {k: aug[k] for k in REQUIRED_FIELDS} if aug else None,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                done += 1

            fout.flush()

            if (event_idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = (done - len(resume_keys)) / elapsed * 3600 if elapsed > 0 else 0
                print(
                    f"  Events: {event_idx+1}/{len(events)} | "
                    f"Records: {done}/{total_candidates} | "
                    f"Rate: {rate:.0f} rec/hr"
                )

    print(f"\nDone. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Score candidate hints (SLM-aligned reward)")
    parser.add_argument("--candidates", default="policy_data/dpo/candidate_hints_train.jsonl")
    parser.add_argument("--output", default="policy_data/dpo/scored_candidates_train.jsonl")
    parser.add_argument("--split", default=None,
                        help="If set, auto-sets candidates and output paths")

    # Backend
    parser.add_argument("--backend", choices=["api", "local"], default="api",
                        help="api: OpenAI-compatible concurrent calls (fast); "
                             "local: local Qwen3-8B inference (slow, offline)")

    # API options
    parser.add_argument("--config", default="configs/api_config.yaml",
                        help="API config file (backend=api)")
    parser.add_argument("--api_workers", type=int, default=20,
                        help="Concurrent API workers per event batch (backend=api)")
    parser.add_argument("--api_max_retries", type=int, default=3)
    parser.add_argument("--api_retry_delay", type=int, default=5)

    # Local options
    parser.add_argument("--llm_model", default="Qwen/Qwen3-8B",
                        help="Local model path (backend=local)")

    # SLM probe (both backends)
    parser.add_argument("--slm_probe", default="slm_outputs/basepack_only/best_model")

    args = parser.parse_args()

    if args.split:
        args.candidates = f"policy_data/dpo/candidate_hints_{args.split}.jsonl"
        args.output = f"policy_data/dpo/scored_candidates_{args.split}.jsonl"

    # Set up LLM backend
    api_client, api_model = None, ""
    api_max_retries, api_retry_delay = args.api_max_retries, args.api_retry_delay

    if args.backend == "api":
        if not _OPENAI_AVAILABLE:
            raise ImportError("pip install openai pyyaml")
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        api_cfg = cfg["api"]
        env_key = (
            os.environ.get("DASHSCOPE_API_KEY", "").strip()
            or os.environ.get("LLM_API_KEY", "").strip()
        )
        if env_key:
            api_cfg["api_key"] = env_key
        if not api_cfg.get("api_key"):
            raise EnvironmentError("API key not set. Export DASHSCOPE_API_KEY.")
        api_client = OpenAI(api_key=api_cfg["api_key"], base_url=api_cfg.get("base_url"))
        api_model = api_cfg["model"]
        api_max_retries = api_cfg.get("max_retries", args.api_max_retries)
        api_retry_delay = api_cfg.get("retry_delay", args.api_retry_delay)
        print(f"Backend: API | model={api_model} | workers={args.api_workers}")
    else:
        print(f"Backend: local | model={args.llm_model}")

    score_split(
        Path(args.candidates), Path(args.output),
        args.backend, args.llm_model, args.slm_probe,
        api_client, api_model, args.api_workers,
        api_max_retries, api_retry_delay,
    )


if __name__ == "__main__":
    main()
