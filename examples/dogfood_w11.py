"""End-to-end dogfood：corpus_quality + successful_profile + profile_resume_gap.

跑一遍真实 LLM，看：
1. 卖课贴是否被 score=0 排除
2. 画像合成是否带证据归属、是否过滤了 marketer
3. Gap 4 桶是否真的把简历差距分对了
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "new_try" / "src"))


# ─── 加载 .env ───────────────────────────────
def _load_env(path: str) -> None:
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        # strip quotes (incl. smart quote)
        v = v.strip()
        for ch in ('"', "'", "”", "“"):
            v = v.strip(ch)
        os.environ.setdefault(k.strip(), v)

_load_env("/Users/huy/new_try/.env")

from offerguide import corpus_quality, interview_corpus
from offerguide.config import Settings
from offerguide.llm import LLMClient
from offerguide.memory import Store
from offerguide.skills import SkillRuntime, discover_skills

# ─── 配置 ───────────────────────────────────
SKILLS_ROOT = Path("/Users/huy/new_try/src/offerguide/skills")
COMPANY = "字节跳动"
ROLE = "AI Agent 后端实习"


# ─── 5 个种子样本 ────────────────────────────
SAMPLES = [
    # (kind_we_expect, source, raw_text, role_hint)
    (
        "marketer",
        "weibo_promo_001",
        """
🔥【字节大厂内推 · 提前批通道】🔥
拿大厂 offer 不再是梦！加我微信 wx-bytedance-2026 进入专属群

✅ 资料包限免: 字节跳动近 3 年 真题 + 面经 + 项目模板, 直接照搬就能拿 offer
✅ 1v1 辅导: 专业老师改简历, 模拟面试, 包过率 95%
✅ 训练营 9 天提升: 算法 + 八股 + 项目辅导, 价格 999 起
✅ 内推码 限量发放, 加微信领取

🎁 仅限今日: 关注公众号「大厂直通」回复"字节"送资料

DM 我领取完整 字节 25-26 校招资料合集 包过保证
        """.strip(),
        ROLE,
    ),
    (
        "offer_post",
        "nowcoder_offer_002",
        """
字节跳动 Seed AI Agent 后端实习 offer 复盘 (2026/04 上岸)

本人: 上海交大计算机硕士 ML 方向, 有美团 ML 平台一段实习 (3 个月, 2025/9-2025/11), 校招前手撕过 leetcode hot100。

简历核心项目:
1. 自主搭建的 LLM 多 agent 评测平台 (LangGraph + DSPy + Pydantic), 收集了 200+ 测试用例 覆盖 ReAct/CoT/Tool-call 三种模式, 在 GAIA benchmark 上跑出 ReAct 0.34 / CoT 0.41 / Tool-call 0.52 的对比数据。
2. RAG 系统 (BGE-large + bce-rerank), 内部知识库 hit@5 0.78。

时间线: 3/15 投递 → 3/22 一面 → 3/27 二面 → 4/3 三面 → 4/8 oc。

一面 (60min, 主管面):
- 介绍项目, 面试官追问 LangGraph state machine 怎么设计, 你的 100 个 agent 并发跑过吗
- 问 GRPO 和 PPO 区别, 为什么不直接用 PPO, 我答 GRPO 不需要 critic, 用 group baseline 估计 advantage
- 手撕反转链表 (leetcode 206), 写 iterative + recursive 两种, 写完讨论复杂度

二面 (45min, 技术 leader):
- 系统设计: 设计支持 1000 QPS 的 agent inference pipeline
- 让我画图, 我提了 batch + speculative decoding + KV cache reuse, 面试官点头
- 问 RAG 怎么 debug retrieval miss, 我答 query rewriting + hybrid search + reranker calibration

三面 (HR + cross-team): 个人定位、长期想做什么, 把项目讲了三遍每次不同视角。

我觉得过的原因:
- 项目深度被多次表扬 (二面面试官说"这个评测平台听起来比我们内部用的还系统")
- 系统设计部分给了具体取舍 (不是只说概念, 提了具体数字 latency target)
- 一面面试官追问时我没装懂的就直接说"这部分我不太了解, 但我可以这么推断..."
        """.strip(),
        ROLE,
    ),
    (
        "interview",
        "nowcoder_intervew_003",
        """
字节 Seed 一面挂了 (2026/04/28)

岗位: AI Agent 后端实习

时长 30min, 几乎全是手撕 + 八股, 没问项目深挖。

题目:
1. 手撕: 给一个数组, 找出第 k 大元素 (leetcode 215)。我用 quick select 写, 写到一半卡了, 面试官提示用 heapq 我改用 heap
2. Transformer attention 缩放因子 √d 的推导, 我答了防止 dot-product 高维饱和, softmax 进入饱和区梯度消失
3. 追问: 如果维度更高怎么办, 我答 LayerNorm + 缩放, 面试官说还有什么, 我没答上来 (其实是 ALiBi 之类)
4. PyTorch 自动求导原理, 计算图怎么构建的
5. 八股: TCP 三次握手, MySQL 索引底层 (B+tree), Redis 数据结构 5 种

二面通知: 没收到 (一周后查 portal 是挂了)。

复盘:
- attention 追问没接住, 应该提 ALiBi / RoPE / sinusoidal 各自的优劣
- 项目方向我有准备但面试官根本没问, 题型是基础八股 + 算法, 不是项目深挖
- 比起 offer 复盘里的同岗位流程, 我这个一面体感是相对偏八股的, 可能是不同面试官
        """.strip(),
        ROLE,
    ),
    (
        "project_share",
        "github_repo_004",
        """
[GitHub Project] LLM-Agent-Eval-Framework

作者本人 拿到字节 Seed AI Agent 后端实习 offer 后整理开源 (2026/04)

核心: 把 200+ 测试用例参数化, 支持 ReAct / CoT / Tool-call 三种 agent 模式对比, 跑在 GAIA / SWE-bench-mini benchmarks 上。

技术栈:
- LangGraph 0.2 (state machine + interrupt)
- DSPy 3.x (prompt 编译 + 评估)
- Pydantic v2 strict (输入输出 schema)
- pytest-bdd (行为驱动测试)

数据 (GAIA Level-1, claude-haiku):
| Mode       | Task Complete | Tool Call Success | Token Cost |
|------------|---------------|-------------------|------------|
| ReAct      | 0.34          | 0.71              | 12k tok    |
| CoT        | 0.41          | n/a               | 8k tok     |
| Tool-call  | 0.52          | 0.83              | 9k tok     |

关键 insight:
- Tool-call 模式比 ReAct 强: ReAct 的 thought→action 链条在多步任务里容易跑偏
- 把 system prompt 里的 example 数从 0 加到 3, completion rate 提升 0.1+, 但是 token cost 翻倍
- 评估指标必须看 `tool_call_success` 不能只看 `task_complete`, 因为有些任务 LLM 编了答案

代码结构:
```
src/agent_eval/
├── modes/        # 三个 agent 实现
├── benchmarks/   # GAIA / SWE-bench-mini 加载器
├── metrics/      # 评估指标
└── runner.py
```
        """.strip(),
        ROLE,
    ),
    (
        "other",
        "manual_paste_005",
        """
请教大家, 字节 AI Agent 实习一般什么时候开放暑期通道?

我是 25 届的, 想问下 2026 暑期实习的字节 Seed AI 团队大概几月开始招? 我看了 portal 没明显公告。

听说他们提前批是滚动招聘, 那是不是随时可以投? 然后简历是不是统一筛过几轮还是单部门筛?

希望有过经验的同学来分享下流程~ 谢谢
        """.strip(),
        ROLE,
    ),
]


def main():
    # ─── Step 0: 验证 LLM ──────────────────────
    settings = Settings.from_env()
    print(f"[ENV] base_url = {settings.deepseek_base_url}")
    print(f"[ENV] api_key = {settings.deepseek_api_key[:10] if settings.deepseek_api_key else None}...")
    print(f"[ENV] model = {settings.default_model}")

    if not settings.deepseek_api_key:
        print("ERROR: no API key in .env (BASE_URL/TOKEN expected)")
        return 1

    llm = LLMClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        default_model="claude-haiku-4-5-20251001",  # cheap/fast for classifier
    )

    # ─── Step 1: 准备 DB ────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name
    print(f"[DB] {db_path}")
    store = Store(db_path)
    store.init_schema()

    # ─── Step 2: 种 5 个样本 ────────────────────
    print("\n========== STEP 2: 种 5 个样本 ==========")
    seeded = []
    for expected_kind, source, raw, role in SAMPLES:
        was_new, ie_id = interview_corpus.insert(
            store, company=COMPANY, raw_text=raw, source=source, role_hint=role
        )
        seeded.append((ie_id, expected_kind, source))
        print(f"  ✓ id={ie_id} expected={expected_kind:14s} source={source} ({len(raw)} 字)")

    # ─── Step 3: 真 LLM 跑质量分类 ──────────────
    print("\n========== STEP 3: corpus_quality.classify_pending (real LLM) ==========")
    print("running...")
    counters = corpus_quality.classify_pending(store, llm=llm, limit=10)
    print(f"counters: {counters}")

    print("\n  per-row verdicts:")
    with store.connect() as c:
        rows = c.execute(
            "SELECT id, content_kind, quality_score, quality_signals_json "
            "FROM interview_experiences ORDER BY id"
        ).fetchall()
    for ie_id, expected, source in seeded:
        match = next((r for r in rows if r[0] == ie_id), None)
        if not match:
            continue
        kind = match[1]
        score = match[2]
        signals = json.loads(match[3]) if match[3] else {}
        rationale = signals.get("rationale", "(none)")
        ok = "✓" if kind == expected else "✗"
        print(f"  {ok} id={ie_id} expected={expected:14s} got={kind:14s} score={score:.2f}")
        print(f"      rationale: {rationale[:90]}")

    # ─── Step 4: fetch high-quality ────────────
    print("\n========== STEP 4: fetch_high_quality 拉合成原料 ==========")
    hq = corpus_quality.fetch_high_quality(
        store, company=COMPANY, role_hint=ROLE, min_score=0.6,
    )
    print(f"  high-quality samples: {len(hq)}")
    for s in hq:
        print(f"    · id={s['id']:2d} kind={s['content_kind']:14s} score={s['quality_score']:.2f} src={s['source']}")

    if not hq:
        print("  ⚠ 0 high-quality, can't synthesize profile. Lowering threshold...")
        hq = corpus_quality.fetch_high_quality(
            store, company=COMPANY, role_hint=ROLE, min_score=0.4,
        )
        print(f"  with min_score=0.4: {len(hq)} samples")

    # ─── Step 5: successful_profile SKILL ──────
    print("\n========== STEP 5: successful_profile SKILL (real LLM) ==========")
    skills = discover_skills(SKILLS_ROOT)
    runtime = SkillRuntime(
        llm=LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model="claude-sonnet-4-6",  # better synthesis
        ),
        store=store,
    )
    sp_spec = next(s for s in skills if s.name == "successful_profile")

    samples_for_skill = [
        {
            "id": s["id"],
            "content_kind": s["content_kind"],
            "raw_text": (s["raw_text"] or "")[:3000],
            "source": s["source"],
            "source_url": s["source_url"] or "",
            "quality_score": s["quality_score"],
        }
        for s in hq
    ]
    print(f"  invoking with {len(samples_for_skill)} samples...")

    sp_result = runtime.invoke(sp_spec, {
        "company": COMPANY,
        "role_hint": ROLE,
        "high_quality_samples_json": json.dumps(samples_for_skill, ensure_ascii=False),
    })

    profile = sp_result.parsed or {}
    print(f"\n  ✓ profile generated (run #{sp_result.skill_run_id})")
    print(f"    company: {profile.get('company')}")
    print(f"    role_focus: {profile.get('role_focus')}")
    print(f"    evidence_count: {profile.get('evidence_count')}")
    print(f"    evidence_kinds: {profile.get('evidence_kinds')}")

    if profile.get("skill_pattern"):
        sk = profile["skill_pattern"]
        print(f"\n    must_have: {sk.get('must_have', [])}")
        print(f"    highly_valued: {sk.get('highly_valued', [])}")
        print(f"    differentiators: {sk.get('differentiators', [])}")

    if profile.get("interview_pattern", {}).get("common_questions"):
        print("\n    interview common_questions:")
        for q in profile["interview_pattern"]["common_questions"]:
            print(f"      · [{q.get('category', '?'):20s}] ev={q.get('evidence_count')} - {q.get('question', '')[:60]}")

    if profile.get("why_they_passed"):
        print("\n    why_they_passed:")
        for r in profile["why_they_passed"]:
            print(f"      · {r}")

    if profile.get("uncertainty_notes"):
        print("\n    uncertainty_notes:")
        for u in profile["uncertainty_notes"]:
            print(f"      ⚠ {u}")

    # ─── Step 6: 合成 resume + Gap SKILL ────────
    SYNTHETIC_RESUME = """
胡阳 上海财经大学 应用统计学硕士 (2027 届毕业) shfu.edu.cn

实习经历:
- 法至科技 NLP 工程师 (2025/09 至今):
  · 用 LangGraph + DSPy 搭建客户合规审查多 agent 评测系统
  · 引入 GAIA-CN 类似 benchmark, 100+ 用例覆盖 ReAct / Tool-call 模式
  · RAG retrieval miss 改进: query rewriting 把 hit@5 从 0.61 提到 0.74

项目:
- RemeDi: 基于 BERT 的医疗文本双流推荐, AUC 提升 0.04 (2024/10)
- Deep Research Agent (开源 200 ⭐): LangGraph + DSPy 研究助手 (2025/09)

技能: Python, PyTorch, LangGraph, DSPy, BGE, Pydantic, pytest

教育背景:
- 上海财经大学 应用统计专硕 (2025/09 - 2027/06)
- 西南财经大学 经济统计本科 (2021/09 - 2025/06)
"""
    print("\n========== STEP 6: profile_resume_gap SKILL (real LLM) ==========")
    gap_spec = next(s for s in skills if s.name == "profile_resume_gap")

    print("  invoking with profile + 合成简历...")
    gr = runtime.invoke(gap_spec, {
        "successful_profile_json": json.dumps(profile, ensure_ascii=False),
        "user_resume": SYNTHETIC_RESUME.strip(),
    })

    gap = gr.parsed or {}
    print(f"  ✓ gap analysis generated (run #{gr.skill_run_id})")

    # 4 桶 + 投递建议
    for bucket_name, key in [
        ("✓ 已具备", "已具备"),
        ("⏱ 短期能补 (≤2 周)", "短期能补 (≤2周)"),
        ("⚠ 短期补不了", "短期补不了"),
        ("✗ 不能编", "不能编"),
    ]:
        items = gap.get(key, [])
        print(f"\n  {bucket_name}: {len(items)} 项")
        for it in items[:3]:
            print(f"    · {it.get('topic', '?')}")
            if "evidence_in_resume" in it:
                print(f"        简历: {it['evidence_in_resume'][:80]}")
            if "concrete_action" in it:
                print(f"        动作: {it['concrete_action'][:120]}")
            if "min_time_to_acquire" in it:
                print(f"        最少周期: {it['min_time_to_acquire']}")
            if "why_unfakeable" in it:
                print(f"        验证: {it['why_unfakeable'][:80]}")

    if gap.get("投递建议"):
        adv = gap["投递建议"]
        print(f"\n  ════ 投递建议 ════")
        print(f"  verdict: {adv.get('verdict')}")
        print(f"  rationale: {adv.get('rationale_chinese')}")
        if adv.get("top_3_pre_apply_actions"):
            print(f"  top 3 actions:")
            for i, a in enumerate(adv["top_3_pre_apply_actions"], 1):
                print(f"    {i}. {a}")

    # ─── 总结 ─────────────────────────────────
    print("\n========== TOTAL TOKEN USAGE ==========")
    with store.connect() as c:
        rows = c.execute(
            "SELECT skill_name, latency_ms, cost_usd, "
            "  json_extract(input_json, '$.company') as co "
            "FROM skill_runs ORDER BY id"
        ).fetchall()
    total_cost = 0.0
    total_lat = 0
    for r in rows:
        total_cost += r[2] or 0
        total_lat += r[1] or 0
        print(f"  · {r[0]:30s} latency={r[1] or 0} ms  cost=${r[2] or 0:.4f}")
    print(f"  TOTAL latency={total_lat} ms  cost=${total_cost:.4f}")

    os.unlink(db_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
