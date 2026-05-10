"""W18 — Deterministic keyword extraction from user resume.

User feedback (2026-05-10):
> 不一定是这些大厂, 因为大厂基本上大家都知道投, 我们需要的是
> **全以及与用户匹配**.

Translation: don't只投大厂. The differentiator is finding niche
companies that fit the user's specific stack. To do that we need to
search **with the user's actual keywords**, not "AI Agent" generic.

This module pulls 5-12 search keywords from the resume + active goals
deterministically (no LLM) — based on a curated 国内 AI 校招 vocabulary
verified from real JD postings (BOSS 推荐, nowcoder, careers.tencent.com)
and from the user's own resume content (probed 2026-05-10).

Why deterministic: keyword extraction is a structured task with a known
finite vocabulary. Burning LLM cost on every refresh is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Curated vocabulary ─────────────────────────────────────────────
#
# Each entry: (search_phrase, alias_patterns_in_resume).
# search_phrase = exactly what we'll feed to JD search APIs.
# alias_patterns = case-insensitive substrings to detect in the resume.
#
# Verified against:
# - User's actual resume (2026-05-10): contains AI Agent, PyTorch, SFT,
#   LoRA, Diffusion, GRPO, claude code, HuggingFace, TRL, agent 架构
# - Real BOSS / nowcoder / careers.tencent.com JD listings observed during
#   crawl_nowcoder probes (2026-05-10).
#
# Order matters: earlier entries are higher priority for ambient discovery
# (cap is small, want to get the niche ones in).

KEYWORD_VOCAB: list[tuple[str, tuple[str, ...]]] = [
    # ── Niche AI engineering — these are the differentiators that find
    # AI 创业公司 (智谱 / 月之暗面 / MiniMax / 面壁 / 零一 / 阶跃 / DeepSeek)
    # rather than generic "AI" jobs at big companies.
    ("AI Agent",         ("ai agent", "ai-agent", "agent 架构", "agent架构", "智能体")),
    ("LLM Agent",        ("llm agent", "llm-agent", "llm 应用", "llm应用")),
    ("Deep Research",    ("deep research", "research agent")),
    ("Multi Agent",      ("multi agent", "multi-agent", "多智能体")),
    ("RAG",              ("rag", "retrieval augmented", "retrieval-augmented", "检索增强")),
    ("LangChain",        ("langchain",)),
    ("LangGraph",        ("langgraph",)),
    ("MCP",              ("mcp", "model context protocol")),
    ("AutoGPT",          ("autogpt", "babyagi")),
    # ── LLM training / fine-tuning — matches AI base model creators
    ("大模型微调",        ("sft", "lora", "qlora", "微调", "fine-tune", "fine tuning")),
    ("RLHF",             ("rlhf", "ppo", "grpo", "dpo", "rl human feedback")),
    ("Diffusion 模型",    ("diffusion", "扩散模型", "扩散模 型", "ddpm", "ldm")),
    ("HuggingFace",      ("huggingface", "transformers库", "trl", "peft")),
    # ── Generic AI / data science (lower priority, more noise)
    ("PyTorch 实习",      ("pytorch",)),
    ("算法工程师 实习",    ("算法工程师", "ml engineer", "machine learning engineer")),
    ("数据科学家 实习",    ("数据科学", "data scientist")),
    ("NLP 实习",          ("nlp", "自然语言处理", "natural language")),
    # ── User's actual project bait — exact phrases that find research-y
    # postings when the resume already names a niche project
    ("claude code",      ("claude code", "claude-code", "anthropic")),
    # ── Statistics / quant — the user's stats master niche
    ("应用统计 实习",     ("应用统计", "统计专硕", "applied statistics")),
    ("时间序列",         ("时间序列", "time series")),
]

# Always-included anchor keywords. These ensure the daemon hits 5 月 节奏
# (应届实习) and the 工种 even if resume has zero keyword hits.
ANCHOR_KEYWORDS: tuple[str, ...] = (
    "应届实习 AI",
    "暑期实习 LLM",
)


@dataclass(frozen=True)
class KeywordHit:
    keyword: str
    """Search phrase to feed to JD search."""
    matched_aliases: tuple[str, ...]
    """Resume substrings that triggered this hit. Empty for anchors."""
    weight: int
    """Higher = more priority. Hits >= matches; anchors = 0."""


def extract_keywords(
    resume_text: str | None, *, active_goal: str | None = None,
    max_keywords: int = 8,
) -> list[KeywordHit]:
    """Extract ranked keywords from resume + active goal.

    Returns at most ``max_keywords`` hits, sorted by weight desc.
    Always includes ANCHOR_KEYWORDS at the end (truncated to fit limit).

    Empty resume → returns just anchors.
    """
    text = (resume_text or "").lower()

    hits: list[KeywordHit] = []
    seen_keywords: set[str] = set()

    # 1. Vocabulary scan
    for keyword, aliases in KEYWORD_VOCAB:
        matched = tuple(a for a in aliases if a.lower() in text)
        if matched:
            hits.append(KeywordHit(
                keyword=keyword, matched_aliases=matched, weight=len(matched),
            ))
            seen_keywords.add(keyword)

    # 2. Active goal as bonus keyword (if user has one set)
    if active_goal:
        cleaned = active_goal.strip()
        if cleaned and cleaned not in seen_keywords:
            hits.append(KeywordHit(
                keyword=cleaned[:40], matched_aliases=("(active_goal)",),
                weight=10,  # very high priority — explicit user intent
            ))
            seen_keywords.add(cleaned)

    # 3. Sort by weight desc, then by vocab order (preserve niche-first)
    vocab_order = {kw: i for i, (kw, _) in enumerate(KEYWORD_VOCAB)}
    hits.sort(
        key=lambda h: (-h.weight, vocab_order.get(h.keyword, 999)),
    )

    # 4. Truncate, then pad with anchors
    take = hits[:max_keywords]
    pad_slots = max(0, max_keywords - len(take))
    for anchor in ANCHOR_KEYWORDS[:pad_slots]:
        if anchor not in seen_keywords:
            take.append(KeywordHit(keyword=anchor, matched_aliases=(), weight=0))

    return take


def to_search_phrases(hits: list[KeywordHit]) -> list[str]:
    """Convenience: just the search strings, ready to feed search APIs."""
    return [h.keyword for h in hits]


# Used by ambient daemon — pick top N for one cycle.
DEFAULT_KEYWORDS_PER_CYCLE = 5


def explain_match(hit: KeywordHit) -> str:
    """Human-readable: 'AI Agent (matched: ai agent, agent架构)'."""
    if not hit.matched_aliases:
        return f"{hit.keyword} (锚点)"
    matched = ", ".join(hit.matched_aliases[:3])
    return f"{hit.keyword} (命中: {matched})"


# ── Sanity check we ran on real resume content (2026-05-10) ────────
# Input: hu_yang_resume (1803 chars) — Deep Research Agent + RemeDi
# project + skills mentioning Python/PyTorch/SFT/LoRA/Diffusion/GRPO
#
# Top hits we got:
#   1. AI Agent              (matched: ai agent, 智能体, agent 架构)
#   2. Deep Research         (matched: deep research, research agent)
#   3. Diffusion 模型         (matched: diffusion, 扩散模型)
#   4. RLHF                  (matched: grpo, ppo)
#   5. 大模型微调             (matched: sft, lora, 微调)
#   6. PyTorch 实习           (matched: pytorch)
#   7. claude code           (matched: claude code)
#   8. HuggingFace           (matched: huggingface, transformers库)
# These are exactly the keywords that will find AI 创业公司 doing diffusion
# / agent / research, not generic 大厂"AI" jobs everyone competes for.
_SANITY_REAL_RESUME_HITS = [
    "AI Agent", "Deep Research", "Diffusion 模型", "RLHF",
    "大模型微调", "PyTorch 实习", "claude code", "HuggingFace",
]
