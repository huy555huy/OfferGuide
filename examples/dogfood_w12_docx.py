"""真 LLM dogfood: 上财简历.docx + 字节 AI Agent JD → tailored.docx"""
import os, sys
def _load_env(path):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1); v = v.strip()
        for ch in ('"', "'", "”", "“"): v = v.strip(ch)
        os.environ[k.strip()] = v
_load_env("/Users/huy/new_try/.env")

sys.path.insert(0, "/Users/huy/new_try/src")
from offerguide.config import Settings
from offerguide.llm import LLMClient
from offerguide.profile import load_resume_pdf
from offerguide.skills.tailor_resume.docx_tailor import tailor_docx

s = Settings.from_env()
print(f"[ENV] base={s.deepseek_base_url}  model={s.default_model}")
print(f"[ENV] api_key={s.deepseek_api_key[:12]}...")
llm = LLMClient(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url, default_model=s.default_model)

profile = load_resume_pdf("/Users/huy/new_try/中文简历.docx")
print(f"[RESUME] {len(profile.raw_resume_text)} 字符")

JD = """字节跳动 - Seed AI Agent 后端实习

【职责】
1. 参与下一代 LLM Agent 系统设计与实现
2. 用 LangGraph / DSPy 等框架搭建复杂 agent workflow + RAG
3. 设计高 QPS agent inference pipeline
4. 评估 + 优化 agent 在 GAIA / SWE-bench 等 benchmark 上的表现

【要求】
- Python + PyTorch / TensorFlow 熟练
- 熟悉 Transformer / Attention / RAG / Agent 框架原理
- 熟悉 LangGraph / LangChain / DSPy 任一
- 算法基础（leetcode hot100 级别）
- 加分: 有开源 agent 项目经验 / RLHF (PPO/GRPO)

【地点】上海 / 北京"""

print(f"\n=== 跑 tailor_docx (Claude Sonnet 30-90 秒) ===")
result = tailor_docx(
    input_path="/Users/huy/new_try/中文简历.docx",
    output_path="/tmp/胡阳_字节AI_Agent_tailored.docx",
    jd_text=JD,
    company="字节跳动",
    role_focus="Seed AI Agent 后端实习",
    master_resume_text=profile.raw_resume_text,
    llm=llm,
)
print(f"\n=== 完 ===")
print(f"output: {result.output_path}")
print(f"summary: {result.summary()}")

print(f"\n=== 跳过的段落 ({len(result.skipped)}) ===")
for p in result.skipped[:8]:
    print(f"  [{p.index:2d}] skip={p.skip_reason:25s} {p.original_text[:50]!r}")

print(f"\n=== 真改写的段落 ({len(result.changes)}) ===")
for p in result.changes:
    print(f"\n  [{p.index:2d}] {p.rationale}")
    print(f"    BEFORE: {p.original_text[:120]!r}")
    print(f"    AFTER:  {p.new_text[:120]!r}")
