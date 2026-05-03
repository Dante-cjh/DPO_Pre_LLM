"""
_policy_input.py - single source of truth for policy_input formatting.

All scripts that build policy_input should import build_policy_input from here,
so SFT training, DPO candidate generation, DPO pair construction, and policy
inference see the same input distribution.
"""

REPLY_TRUNC = 150
STANCES = ["support", "deny", "query", "neutral"]


def build_policy_input(event: dict) -> str:
    """
    Input: one basepack_v2 event dict with source_text / selected_replies / stats / stance_dist.
    Output: user input text for the policy model.
    """
    source_text = event.get("source_text", "")
    replies = event.get("selected_replies", []) or []

    lines = ["SOURCE:", source_text, "", "REPLIES:"]
    for i, r in enumerate(replies, 1):
        # selected_replies in v2 is dict {"text", "stance"}; keep compatibility with v1 strings.
        if isinstance(r, dict):
            text = r.get("text", "")
        else:
            text = str(r)
        lines.append(f"{i}. {text[:REPLY_TRUNC]}")

    stats = event.get("stats", {}) or {}
    lines.append("")
    lines.append(
        f"STATS:\nreply_count={stats.get('num_replies', 0)} | "
        f"depth={stats.get('max_depth', 0)} | "
        f"branches={stats.get('num_branches', 0)}"
    )

    stance_dist = event.get("stance_dist", {}) or {}
    if stance_dist:
        dist_str = " | ".join(
            f"{s}={stance_dist.get(s, 0.0)}" for s in STANCES
        )
        lines.append(f"STANCE_DIST: {dist_str}")
    return "\n".join(lines)
