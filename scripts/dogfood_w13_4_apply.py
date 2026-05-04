"""W13.4 dogfood — apply_assistant generates a real package for a real job."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ENV = Path(__file__).parent.parent / ".env"
if ENV.exists():
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from offerguide.llm import LLMClient  # noqa: E402
from offerguide.memory import Store  # noqa: E402
from offerguide.profile import load_resume_pdf  # noqa: E402
from offerguide.skills import SkillRuntime, discover_skills  # noqa: E402
from offerguide.skills.apply_assistant.helpers import ApplyPackage  # noqa: E402

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


def main():
    db_path = "/tmp/offerguide_w13_4_dogfood.db"
    if Path(db_path).exists():
        Path(db_path).unlink()
    store = Store(db_path)
    store.init_schema()

    api_key = os.environ.get("OFFERGUIDE_LLM_API_KEY")
    base_url = os.environ.get("OFFERGUIDE_LLM_BASE_URL")
    model = os.environ.get("OFFERGUIDE_LLM_MODEL")
    if not api_key:
        print("[err] no API key"); sys.exit(1)

    llm = LLMClient(api_key=api_key, base_url=base_url, default_model=model)
    skills = discover_skills(SKILLS_ROOT)
    runtime = SkillRuntime(llm=llm, store=store)

    spec = next(s for s in skills if s.name == "apply_assistant")

    resume_path = os.environ.get("OFFERGUIDE_RESUME_PDF")
    if not resume_path or not Path(resume_path).exists():
        print(f"[err] OFFERGUIDE_RESUME_PDF not set / missing")
        sys.exit(1)
    profile = load_resume_pdf(Path(resume_path))
    print(f"[resume] loaded {len(profile.raw_resume_text)} chars")

    job_text = (
        "字节跳动 · AI Lab · LLM Agent 工程师实习\n"
        "工作地点: 北京·海淀\n\n"
        "岗位职责:\n"
        "1. 参与字节内部 AI Agent 平台核心模块开发\n"
        "2. 设计 LLM agent runtime: tool registry / 长期记忆 / RAG\n"
        "3. 探索 GEPA / DSPy / Constitutional AI 等前沿方法工程化\n\n"
        "要求:\n"
        "1. 熟练 Python, 有 LangGraph 或类似 agent 框架经验\n"
        "2. 熟悉 LLM 应用开发: prompt engineering / RAG / tool calling\n"
        "3. 数学/统计基础, 能阅读论文实现\n"
        "4. 在校生硕士/博士, 2026/27 届优先\n"
    )

    print(f"\n[invoke] apply_assistant on 字节跳动·LLM Agent 实习...")
    print("=" * 70)
    result = runtime.invoke(spec, {
        "company": "字节跳动",
        "role_focus": "LLM Agent 工程师实习",
        "job_text": job_text,
        "user_profile": profile.raw_resume_text,
    })
    print(f"latency: {result.latency_ms}ms · run_id: {result.skill_run_id}")
    print()

    if result.parsed is None:
        print("[err] LLM didn't return valid JSON. Raw text:")
        print(result.raw_text[:1000])
        sys.exit(1)

    try:
        pkg = ApplyPackage(**result.parsed)
    except Exception as e:
        print(f"[err] schema validation failed: {e}")
        print("Raw parsed:")
        print(json.dumps(result.parsed, ensure_ascii=False, indent=2)[:2000])
        sys.exit(1)

    print(f"\n{'=' * 70}\nPACKAGE\n{'=' * 70}")
    print(f"\n## 第一句话 ({pkg.self_intro_snippet.platform_hint}):")
    print(f"  {pkg.self_intro_snippet.text}")
    print(f"  rationale: {pkg.self_intro_snippet.rationale}")

    print(f"\n## Q/A 模板 ({len(pkg.qa_templates)} 条):")
    for i, qa in enumerate(pkg.qa_templates, 1):
        print(f"\n  {i}. [{qa.category}] {qa.question} (个性化 {qa.personalization_score:.2f})")
        print(f"     答: {qa.answer[:200]}{'...' if len(qa.answer) > 200 else ''}")
        if qa.anti_patterns:
            print(f"     反模式: {qa.anti_patterns}")

    print(f"\n## 投递策略:")
    print(f"  最佳时间窗: {pkg.submission_strategy.best_time_window}")
    print(f"  预期回复: {pkg.submission_strategy.expected_response_window_days} 天")
    print(f"  没回怎么办: {pkg.submission_strategy.follow_up_plan}")
    if pkg.submission_strategy.platform_specific_tips:
        print(f"  平台贴士:")
        for tip in pkg.submission_strategy.platform_specific_tips:
            print(f"    - {tip}")

    print(f"\n## Pre-submit checklist ({len(pkg.pre_submit_checklist)} 条):")
    for item in pkg.pre_submit_checklist:
        print(f"  ☐ {item}")

    if pkg.skip_reasons:
        print(f"\n## ⚠ 不建议投: {pkg.skip_reasons}")
    else:
        print(f"\n## ✓ 推荐投递")

    print(f"\nconfidence: {pkg.confidence:.2f}")


if __name__ == "__main__":
    main()
