"""Source-aware application plan for semi-automated submission.

This module is deliberately deterministic. The LLM writes the personalized
words; this layer answers the operational question: where does the user apply,
what fields/materials should be ready, and what happens after they click send.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class CopyField:
    label: str
    value_hint: str
    source: str
    copyable: bool = False


@dataclass(frozen=True)
class ApplicationPlan:
    platform: str
    platform_label: str
    apply_url: str | None
    action_label: str
    channel_note: str
    steps: list[str] = field(default_factory=list)
    fields: list[CopyField] = field(default_factory=list)
    material_checklist: list[str] = field(default_factory=list)
    post_apply_actions: list[str] = field(default_factory=list)
    needs_user_send: bool = True
    verified_source: bool = False
    evidence_url: str | None = None


def build_application_plan(job: dict[str, Any]) -> ApplicationPlan:
    """Build a paste-and-click application plan from a jobs row-like dict."""
    source = str(job.get("source") or "").lower()
    url = _clean_url(job.get("url"))
    extras = _load_extras(job)
    verified_source = bool(extras.get("source_verified"))
    evidence_url = _clean_url(extras.get("detail_evidence_url") or extras.get("evidence_url"))
    host = urlparse(url).netloc.lower() if url else ""
    title = str(job.get("title") or "目标岗位")
    company = str(job.get("company") or "目标公司")

    if _is_boss(source, host):
        return ApplicationPlan(
            platform="boss_zhipin",
            platform_label="BOSS 直聘",
            apply_url=url,
            action_label="打开 BOSS 沟通",
            channel_note="OfferGuide 准备评分、开场白和材料；用户在 BOSS 页面审核后自己点发送。",
            steps=[
                "打开原 JD，确认岗位仍在招且公司/地点/薪资没有变。",
                "点击立即沟通或继续沟通，粘贴自我介绍话术。",
                "选择或上传针对这家公司微调过的 PDF 简历。",
                "发送前看一眼称呼、岗位名和附件，确认无误后由你点击发送。",
            ],
            fields=[
                CopyField("沟通开场白", "使用下方“自我介绍话术”", "apply_assistant", True),
                CopyField("目标岗位", f"{company} · {title}", "JD", True),
                CopyField("到岗/实习时间", "按你的真实可到岗时间填写；不确定就先不在第一句话里写", "用户确认"),
                CopyField("附件简历", _resume_filename_hint(company, title), "tailor_resume"),
            ],
            material_checklist=[
                *_common_materials(company, title),
                "BOSS 在线简历和附件简历信息一致，最近项目排在前面。",
                "开场白不要问薪资、转正或 HC，先让 HR 愿意打开简历。",
            ],
            post_apply_actions=_common_post_actions(),
            verified_source=verified_source,
            evidence_url=evidence_url,
        )

    if _is_nowcoder(source, host):
        return ApplicationPlan(
            platform="nowcoder",
            platform_label="牛客",
            apply_url=url,
            action_label="打开牛客投递",
            channel_note="牛客可通过公开 JD 入库；OfferGuide 准备投递确认、私信话术和后续面经备战。",
            steps=[
                "打开牛客 JD，确认届别、投递截止时间和投递按钮状态。",
                "按牛客提示选择在线简历或附件简历。",
                "如果页面支持和招聘者沟通，粘贴自我介绍话术并删掉不适合牛客语境的称呼。",
                "投递成功后回到 OfferGuide 点“我投了”，系统会进入投后备战。",
            ],
            fields=[
                CopyField("牛客打招呼", "使用下方“自我介绍话术”", "apply_assistant", True),
                CopyField("目标岗位", f"{company} · {title}", "JD", True),
                CopyField("简历附件", _resume_filename_hint(company, title), "tailor_resume"),
                CopyField("届别/毕业时间", "按学信/简历一致填写", "用户资料"),
            ],
            material_checklist=[
                *_common_materials(company, title),
                "确认牛客在线简历已更新到最新版本。",
                "如果 JD 有截止日期，投递后当天就开始准备笔试/面试题。",
            ],
            post_apply_actions=_common_post_actions(),
            verified_source=verified_source,
            evidence_url=evidence_url,
        )

    if url and not url.startswith("paste://"):
        verified_prefix = (
            "这条 JD 来自已验证的官方招聘源；OfferGuide 只预填可复制材料，最终提交仍由用户确认。"
            if verified_source
            else "入口来自已入库 URL；OfferGuide 不假设官网表单长什么样，打开页面后逐项核验。"
        )
        return ApplicationPlan(
            platform="official_site",
            platform_label="官网 / ATS 网申" + (" · 已核验来源" if verified_source else ""),
            apply_url=url,
            action_label="打开官网网申",
            channel_note=verified_prefix,
            steps=[
                "打开官网/ATS 页面，确认岗位名、城市和截止时间。",
                "遇到简历解析页先上传微调版 PDF，再逐项校对解析出来的学校、经历和项目。",
                "遇到开放题时使用下方 QA 模板，按页面字数限制裁剪。",
                "最终提交前截屏或记录申请编号，回到 OfferGuide 点“我投了”。",
            ],
            fields=[
                CopyField("申请岗位", f"{company} · {title}", "JD", True),
                CopyField("姓名 / 手机 / 邮箱", "从简历或个人资料复制，保持全站一致", "用户资料"),
                CopyField("学校 / 专业 / 学历 / 毕业时间", "按简历一致填写，避免 ATS 解析冲突", "用户资料"),
                CopyField("项目经历摘要", "优先复制下方 QA 中最贴 JD 的项目段落", "apply_assistant", True),
                CopyField("求职动机 / Why us", "使用下方 motivation 类 QA 答案", "apply_assistant", True),
                CopyField("可到岗时间 / 每周天数", "只填真实可承诺的时间", "用户确认"),
                CopyField("附件简历文件名", _resume_filename_hint(company, title), "tailor_resume", True),
            ],
            material_checklist=[
                *_common_materials(company, title),
                "准备一份可上传 PDF，文件大小尽量小于 5MB。",
                "如果官网要求英文字段，先把中文答案人工确认后再翻译，别直接整段机翻。",
            ],
            post_apply_actions=_common_post_actions(),
            verified_source=verified_source,
            evidence_url=evidence_url,
        )

    return ApplicationPlan(
        platform="unknown",
        platform_label="未识别渠道",
        apply_url=url,
        action_label="补充投递链接",
        channel_note="这个岗位还缺真实投递入口；先准备材料，投递前需要补官网/BOSS/牛客 URL。",
        steps=[
            "先用公司名和岗位名搜索官网/BOSS/牛客的真实 JD。",
            "找到入口后把 URL 回填或重新评估，OfferGuide 会按渠道生成更具体的步骤。",
            "如果只能邮件投递，把自我介绍话术改成邮件正文第一段。",
        ],
        fields=[
            CopyField("目标岗位", f"{company} · {title}", "JD", True),
            CopyField("附件简历文件名", _resume_filename_hint(company, title), "tailor_resume", True),
            CopyField("邮件/表单正文", "使用自我介绍话术 + motivation QA 组合", "apply_assistant"),
        ],
        material_checklist=_common_materials(company, title),
        post_apply_actions=_common_post_actions(),
        verified_source=verified_source,
        evidence_url=evidence_url,
    )


def _clean_url(value: Any) -> str | None:
    if not value:
        return None
    url = str(value).strip()
    return url or None


def _load_extras(job: dict[str, Any]) -> dict[str, Any]:
    raw = job.get("extras")
    if isinstance(raw, dict):
        return raw
    raw = job.get("extras_json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _is_boss(source: str, host: str) -> bool:
    return source.startswith("boss") or "zhipin.com" in host


def _is_nowcoder(source: str, host: str) -> bool:
    return source == "nowcoder" or "nowcoder.com" in host


def _resume_filename_hint(company: str, title: str) -> str:
    compact_company = _compact(company) or "公司"
    compact_title = _compact(title) or "岗位"
    return f"<姓名>_{compact_company}_{compact_title}_简历.pdf"


def _compact(text: str) -> str:
    return "".join(ch for ch in text.strip() if not ch.isspace() and ch not in "/\\:：")


def _common_materials(company: str, title: str) -> list[str]:
    return [
        f"确认是否接受这次针对 {company} · {title} 的简历微调；不接受就使用 master 简历。",
        "所有可复制答案只作为草稿，提交前按页面字数和真实情况改一遍。",
        "投递前确认手机号、邮箱、毕业时间和可到岗时间没有互相打架。",
    ]


def _common_post_actions() -> list[str]:
    return [
        "投递后立刻在 OfferGuide 标记“我投了”，让系统开始跟踪沉默期。",
        "打开投后备战包，先看该公司/岗位最可能问的 5-8 个问题。",
        "5-7 天没有回应时再跟进，不发“在吗”，补充一个和 JD 对齐的新信息。",
    ]
