"""
policy_input_utils.py
统一 prompt policy 的输入格式，供 02 / 03 / 05 / 06 共用。

设计原则（修复"训推漂移"）：
  1. 所有阶段（SFT 训练、DPO 候选生成、DPO pair 重建、推理出 hint）
     都从 BasePack-v2 文件读，且都通过本模块构造 policy_input。
  2. 回复统一截断到 REPLY_CHAR_LIMIT 字符。
  3. STATS / STANCE_DIST 始终拼接（v2 一定有 stance_dist 字段）。
  4. selected_replies 兼容 dict({"text", "stance"}) 和 str 两种格式。

INSTRUCTION 也在此集中维护，避免多处复制粘贴漂移。
"""

from __future__ import annotations

from pathlib import Path

REPLY_CHAR_LIMIT = 150
STANCES = ("support", "deny", "query", "neutral")

INSTRUCTION = (
    "Generate a concise focus_hint for LLM-assisted rumor analysis. "
    "Return only a JSON object with the key focus_hint."
)

FALLBACK_HINT = (
    "Focus on the central claim, supporting and refuting replies, conflict among replies, "
    "source grounding, and whether the discussion provides concrete evidence or only emotional reactions."
)


def _reply_text(reply) -> str:
    if isinstance(reply, dict):
        return str(reply.get("text", "")).strip()
    return str(reply).strip()


def build_policy_input(event: dict, reply_char_limit: int = REPLY_CHAR_LIMIT) -> str:
    """构造 prompt policy 的输入文本（SOURCE / REPLIES / STATS / STANCE_DIST）。

    event 应来自 BasePack-v2，需包含 source_text / selected_replies / stats / stance_dist。
    若是 v1 旧文件则 stance_dist 缺失会被忽略（向后兼容，但不推荐）。
    """
    source_text = event.get("source_text", "")
    replies = event.get("selected_replies", []) or []

    lines = ["SOURCE:", source_text, "", "REPLIES:"]
    for i, r in enumerate(replies, 1):
        text = _reply_text(r)
        if reply_char_limit and len(text) > reply_char_limit:
            text = text[:reply_char_limit]
        lines.append(f"{i}. {text}")

    stats = event.get("stats", {}) or {}
    stance_dist = event.get("stance_dist", {}) or {}

    lines.append("")
    lines.append(
        f"STATS:\nreply_count={stats.get('num_replies', 0)} | "
        f"depth={stats.get('max_depth', 0)} | "
        f"branches={stats.get('num_branches', 0)}"
    )
    if stance_dist:
        dist_str = " | ".join(
            f"{s}={stance_dist.get(s, 0.0)}" for s in STANCES
        )
        lines.append(f"STANCE_DIST: {dist_str}")
    return "\n".join(lines)


def basepack_v2_path(split: str, basepack_dir: str = "basepack_v2") -> Path:
    """统一拼 v2 basepack 路径（split ∈ {train,val,test}）。"""
    return Path(basepack_dir) / f"basepack_{split}.jsonl"
