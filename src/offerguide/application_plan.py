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

    # W19+ — host-based 大厂细化 (each company has known login flow + 内推
    # mechanism + 简历上传方式; generic ATS message不够). Host check covers
    # both verified-source ingest paths AND user-pasted URLs.
    if _is_tencent(source, host):
        return _tencent_plan(
            url=url, company=company, title=title,
            verified_source=verified_source, evidence_url=evidence_url,
            is_social=("careers.tencent.com" in host or source == "tencent_social"),
        )
    if _is_baidu(source, host):
        return _baidu_plan(
            url=url, company=company, title=title,
            verified_source=verified_source, evidence_url=evidence_url,
            is_intern=(source == "baidu_intern" or (url is not None and "INTERN" in url)),
        )
    if _is_bytedance(source, host):
        return _bytedance_plan(
            url=url, company=company, title=title,
            verified_source=verified_source, evidence_url=evidence_url,
        )
    if _is_alibaba(source, host):
        return _alibaba_plan(
            url=url, company=company, title=title,
            verified_source=verified_source, evidence_url=evidence_url,
        )
    if _is_meituan(source, host):
        return _meituan_plan(
            url=url, company=company, title=title,
            verified_source=verified_source, evidence_url=evidence_url,
        )
    if _is_mokahr(source, host):
        return _mokahr_plan(
            url=url, company=company, title=title,
            verified_source=verified_source, evidence_url=evidence_url,
        )

    # W19+ — agent_search source needs explicit 'verify first' framing.
    # The job came from LLM web search, not a verified API. User must
    # check the URL still loads + 公司是否真在招前再花时间投.
    if source == "agent_search":
        return _agent_search_plan(
            url=url, company=company, title=title,
            verified_source=False, evidence_url=evidence_url,
        )

    # W20 — 实习僧 (shixiseng.com) 实习专用聚合.
    # 入口在 shixiseng 站内: 用户登录后按"投递" → HR 收简历 (类似牛客).
    # 不去原公司 ATS, 但 HR 看完会 follow up 微信加好友 / 邮件.
    if source == "shixiseng" or "shixiseng.com" in host:
        return _shixiseng_plan(
            url=url, company=company, title=title,
            verified_source=verified_source, evidence_url=evidence_url,
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


def _is_tencent(source: str, host: str) -> bool:
    return (
        source in ("tencent_campus", "tencent_social")
        or "join.qq.com" in host
        or "careers.tencent.com" in host
    )


def _is_baidu(source: str, host: str) -> bool:
    return (
        source in ("baidu_campus", "baidu_intern")
        or "talent.baidu.com" in host
    )


def _is_bytedance(source: str, host: str) -> bool:
    return (
        source.startswith("bytedance")
        or source == "bytedance_jobs"  # W19+ source name
        or "jobs.bytedance.com" in host
    )


def _is_alibaba(source: str, host: str) -> bool:
    return (
        source.startswith("alibaba")
        or "talent.alibaba.com" in host
        or "campus-talent.alibaba.com" in host  # W19+ via 0voice repo
    )


def _is_mokahr(source: str, host: str) -> bool:
    """W19+ — 北森 SaaS app.mokahr.com used by many 中厂/独角兽 (vendored ATS).
    Worth a dedicated plan since the ATS form pattern is consistent across
    its 6000+ enterprise clients."""
    return "mokahr.com" in host


def _is_meituan(source: str, host: str) -> bool:
    return source.startswith("meituan") or "zhaopin.meituan.com" in host


# ── Per-company plan builders ─────────────────────────────────────────


def _tencent_plan(
    *, url: str | None, company: str, title: str,
    verified_source: bool, evidence_url: str | None, is_social: bool,
) -> ApplicationPlan:
    """腾讯校招 (join.qq.com) + 社招 (careers.tencent.com)."""
    if is_social:
        # 应届生不该投社招 — banner warn
        return ApplicationPlan(
            platform="tencent_social",
            platform_label="腾讯社招 · ⚠ 应届生慎投",
            apply_url=url, action_label="打开腾讯社招页",
            channel_note="社招岗位通常要 1-3 年工作经验。应届生投简历多被卡在第一关；想试可投, 但别耗时间补答。",
            steps=[
                "看 JD 要求工作年限、岗位描述里「3+ years」等字眼。如果非常硬就 skip。",
                "QQ / 微信扫码登录 careers.tencent.com。",
                "上传 PDF 简历, 校对 ATS 解析出的字段。",
                "投递后回 OfferGuide 标记, 系统帮你跟踪 2 周静默期。",
            ],
            fields=_tencent_fields(company, title),
            material_checklist=[
                *_common_materials(company, title),
                "腾讯社招简历重点写「实战项目与商业 outcome」, 不要堆课程证书。",
                "如果有大厂背景或开源 contribution, 放在最前面。",
            ],
            post_apply_actions=_common_post_actions(),
            verified_source=verified_source, evidence_url=evidence_url,
        )
    return ApplicationPlan(
        platform="tencent_campus",
        platform_label="腾讯校招 · join.qq.com" + (" · 已核验" if verified_source else ""),
        apply_url=url, action_label="打开腾讯校招页",
        channel_note="腾讯校招走 join.qq.com 一站式 ATS, 微信扫码登录, 一份在线简历能投多个岗位。强烈建议先找到内推码再投 — 内推走 fast track。",
        steps=[
            "微信扫码登录 join.qq.com (没账号会自动注册)。",
            "完善在线简历或上传 PDF — 第一次会有引导, 别跳过「项目经历」字段。",
            "**输入内推码** (强烈建议, 牛客 / 知乎 / 微信群有现成的)。",
            "选目标岗位, 应届实习类型确认「可全职 + 可转正」选项。",
            "投递后立即回 OfferGuide 标「我投了」, 系统跟踪 2 周静默期。",
        ],
        fields=[
            CopyField("申请岗位", f"{company} · {title}", "JD", True),
            CopyField("内推码 (推荐)", "去牛客搜 \"腾讯 2026 内推码\" 或问学长学姐", "用户准备", True),
            CopyField("在线简历内容", "粘 master_resume 完整版, ATS 不限字数", "用户资料"),
            CopyField("项目经历", "用下方 QA 中 Project 类回答", "apply_assistant", True),
            CopyField("求职动机", "用下方 motivation 类回答", "apply_assistant", True),
            CopyField("可到岗时间", "实习开放从入职日 + 实习时长 (>=3 个月通常)", "用户确认"),
            CopyField("附件 PDF 文件名", _resume_filename_hint(company, title), "tailor_resume", True),
        ],
        material_checklist=[
            *_common_materials(company, title),
            "腾讯 ATS 不限简历字数, 在线简历最好填全, 别只放 PDF。",
            "投递后 14-21 天没回应再考虑跟进, 腾讯 HR 节奏慢。",
        ],
        post_apply_actions=[
            "投后 24h 内可加 BG 内推人微信问声「我投了 X 岗」, 礼貌报备。",
            "看牛客「腾讯 2026 校招」标签每天 1 次, 留意笔试通知。",
            *_common_post_actions(),
        ],
        verified_source=verified_source, evidence_url=evidence_url,
    )


def _baidu_plan(
    *, url: str | None, company: str, title: str,
    verified_source: bool, evidence_url: str | None, is_intern: bool,
) -> ApplicationPlan:
    """百度校招 (talent.baidu.com) — GRADUATE 校招 + INTERN 实习两个 entry."""
    intern_label = " · 暑期/日常实习" if is_intern else " · 校招正式"
    return ApplicationPlan(
        platform="baidu_" + ("intern" if is_intern else "campus"),
        platform_label="百度" + intern_label + (" · 已核验" if verified_source else ""),
        apply_url=url, action_label="打开百度招聘页",
        channel_note=(
            "百度走 talent.baidu.com 自研 ATS, 百度账号登录。"
            + ("实习项目分暑期 / 日常 / AIDU 三种, 看岗位名称里的项目类型。" if is_intern else "")
            + " 内推码可选, 不强求。"
        ),
        steps=[
            "百度账号登录 talent.baidu.com (有百度网盘账号即可)。",
            "上传 PDF 简历, 百度 ATS 解析比较严格 — 学校 / 专业 / 毕业时间一定校对。",
            ("实习投递时确认「周到岗天数」和「实习时长」 — 暑期项目一般要求 3-4 月以上。"
                if is_intern else "校招岗位选「毕业入职」时间, 通常是毕业当年 7 月。"),
            "回答开放题 (动机 / 项目深挖) 时用下方 QA 模板。",
            "投递后回 OfferGuide 标「我投了」。",
        ],
        fields=[
            CopyField("申请岗位", f"{company} · {title}", "JD", True),
            CopyField("内推码 (可选)", "去牛客搜 \"百度 2026 内推\"", "用户准备"),
            CopyField("学校/专业/毕业时间", "和简历完全一致, ATS 解析不通过会卡", "用户资料"),
            CopyField("项目深挖", "用下方 QA 中 project_deep_dive 类答案", "apply_assistant", True),
            CopyField("可实习时长", "诚实填; 百度筛 3+ 月", "用户确认") if is_intern else CopyField("毕业时间", "按学位证书时间", "用户资料"),
            CopyField("附件 PDF 文件名", _resume_filename_hint(company, title), "tailor_resume", True),
        ],
        material_checklist=[
            *_common_materials(company, title),
            "百度 PDF 只支持 5MB 以内, 字体内嵌避免 ATS 解析乱码。",
            "项目描述把 LLM / Agent / 检索 等核心词写在前 2 句, ATS 会按关键词扫。",
        ],
        post_apply_actions=_common_post_actions(),
        verified_source=verified_source, evidence_url=evidence_url,
    )


def _bytedance_plan(
    *, url: str | None, company: str, title: str,
    verified_source: bool, evidence_url: str | None,
) -> ApplicationPlan:
    """字节跳动 (jobs.bytedance.com) — 飞书 People 体系."""
    return ApplicationPlan(
        platform="bytedance",
        platform_label="字节跳动 · jobs.bytedance.com",
        apply_url=url, action_label="打开字节招聘页",
        channel_note=(
            "字节走自研飞书 People ATS, 飞书账号登录。**强烈建议先拿内推码** — "
            "内推流程能跳过简历筛, hit rate 提升 3-5x (业内公认)。"
        ),
        steps=[
            "**先去找内推码** (优先级最高): 牛客 \"字节 2026 内推\"、微信「字节内推群」、"
            "知乎专栏。没内推码 hit rate 极低。",
            "飞书扫码登录 jobs.bytedance.com (没飞书账号扫码会自动建)。",
            "上传 PDF 简历, 字节 ATS 解析 OK; 在线简历也要填关键字段。",
            "投递时填内推码字段 — 如果有, 这一步必填。",
            "选岗位 + 城市 (字节多城市可选, 实习一般北京/上海/杭州/深圳)。",
            "投递后立刻在 OfferGuide 标「我投了」。",
        ],
        fields=[
            CopyField("申请岗位", f"{company} · {title}", "JD", True),
            CopyField("⭐ 内推码 (强烈推荐)", "牛客 / 知乎 / 微信群里随便搜 \"字节 2026 内推码\"", "用户准备", True),
            CopyField("学校 / 专业 / 学历 / 毕业时间", "和简历完全一致", "用户资料"),
            CopyField("项目经历", "用下方 QA 中 project 类答案, 把 AI Agent / LLM 核心词放最前", "apply_assistant", True),
            CopyField("可到岗 + 实习时长", "字节实习要求 >= 3 个月, 周到岗 >= 3 天", "用户确认"),
            CopyField("附件 PDF 文件名", _resume_filename_hint(company, title), "tailor_resume", True),
        ],
        material_checklist=[
            *_common_materials(company, title),
            "字节内推码进流程后 7 天内一般有 HR 触达, 没消息可在牛客找内推人补一句。",
            "字节 AI 团队偏好 LLM/Agent/RL 等深技术词, 简历项目描述往这上面靠。",
        ],
        post_apply_actions=[
            "投后 7 天没消息 → 在牛客「字节直聊」板块搜内推人状态。",
            *_common_post_actions(),
        ],
        verified_source=verified_source, evidence_url=evidence_url,
    )


def _alibaba_plan(
    *, url: str | None, company: str, title: str,
    verified_source: bool, evidence_url: str | None,
) -> ApplicationPlan:
    """阿里巴巴 (talent.alibaba.com) — 北森 ATS."""
    return ApplicationPlan(
        platform="alibaba",
        platform_label="阿里巴巴 · talent.alibaba.com",
        apply_url=url, action_label="打开阿里招聘页",
        channel_note=(
            "阿里走北森 ATS (国内主流招聘系统). 淘宝/支付宝账号登录。"
            "阿里实习节奏: 5-7 月开放, 入职后 9-10 月评 offer。内推码可选但能加快流程。"
        ),
        steps=[
            "淘宝 / 支付宝账号登录 talent.alibaba.com。",
            "上传 PDF 简历或填在线简历; 北森 ATS 解析 ok。",
            "回答开放题 (动机 / 项目 / 弱点) 时用下方 QA 模板。",
            "选投递岗位 + 城市 (杭州为主, 北京 / 上海 / 深圳也有 AI 团队)。",
            "投递后立刻在 OfferGuide 标「我投了」。",
        ],
        fields=[
            CopyField("申请岗位", f"{company} · {title}", "JD", True),
            CopyField("内推码 (可选)", "去牛客搜 \"阿里 2026 内推\"", "用户准备"),
            CopyField("学校/专业", "和简历一致", "用户资料"),
            CopyField("项目深挖", "用下方 QA 中 project_deep_dive 类答案", "apply_assistant", True),
            CopyField("求职动机 / 为什么阿里", "用下方 motivation 类答案, 提具体业务线", "apply_assistant", True),
            CopyField("附件 PDF 文件名", _resume_filename_hint(company, title), "tailor_resume", True),
        ],
        material_checklist=[
            *_common_materials(company, title),
            "阿里 AI 业务线分散 (通义 / 达摩院 / 阿里云 / 蚂蚁), 看清是哪条线再写动机。",
        ],
        post_apply_actions=_common_post_actions(),
        verified_source=verified_source, evidence_url=evidence_url,
    )


def _meituan_plan(
    *, url: str | None, company: str, title: str,
    verified_source: bool, evidence_url: str | None,
) -> ApplicationPlan:
    """美团 (zhaopin.meituan.com) — 北森 ATS, 强登录墙."""
    return ApplicationPlan(
        platform="meituan",
        platform_label="美团 · zhaopin.meituan.com · ⚠ 需登录看完整 JD",
        apply_url=url, action_label="打开美团招聘页",
        channel_note=(
            "美团走北森 ATS, 强登录墙 — 不登录连完整 JD 都看不全。"
            "美团账号登录 (没账号扫码自动注册)。"
        ),
        steps=[
            "美团 App 扫码登录 zhaopin.meituan.com, 看完整 JD 详情。",
            "上传 PDF 简历, 北森 ATS 字段解析, 校对学校 / 专业。",
            "回答开放题用下方 QA 模板。",
            "投递后立刻在 OfferGuide 标「我投了」。",
        ],
        fields=[
            CopyField("申请岗位", f"{company} · {title}", "JD", True),
            CopyField("学校/专业", "和简历一致", "用户资料"),
            CopyField("项目经历摘要", "用下方 QA 中 project 类答案", "apply_assistant", True),
            CopyField("附件 PDF 文件名", _resume_filename_hint(company, title), "tailor_resume", True),
        ],
        material_checklist=_common_materials(company, title),
        post_apply_actions=_common_post_actions(),
        verified_source=verified_source, evidence_url=evidence_url,
    )


def _mokahr_plan(
    *, url: str | None, company: str, title: str,
    verified_source: bool, evidence_url: str | None,
) -> ApplicationPlan:
    """W19+ — 北森 SaaS app.mokahr.com plan.

    北森服务 6000+ 中大型企业 (含商汤、明略、第四范式等独角兽), ATS
    UI/字段套路一致, 一份步骤覆盖一大批公司. 0voice repo 的不少 link
    指向这。
    """
    return ApplicationPlan(
        platform="mokahr",
        platform_label=f"{company} · 北森 SaaS",
        apply_url=url, action_label="打开北森网申",
        channel_note=(
            "这家公司用北森 SaaS ATS (国内主流招聘系统, 6000+ 企业在用)。"
            "一般支持微信/手机号注册账号, 简历上传 + 在线编辑两种, ATS 解析较严格。"
        ),
        steps=[
            "微信扫码或手机号注册 app.mokahr.com 账号 (一次注册可投多家)。",
            "上传 PDF 简历, 北森 ATS 解析校对学校 / 专业 / 毕业时间。",
            "回答开放题 (动机 / 项目 / 实习时长) 用下方 QA 模板。",
            "投递后立刻在 OfferGuide 标「我投了」。",
        ],
        fields=[
            CopyField("申请岗位", f"{company} · {title}", "JD", True),
            CopyField("学校 / 专业 / 毕业时间", "和简历完全一致, ATS 解析严格", "用户资料"),
            CopyField("项目经历摘要", "用下方 QA 中 project 类答案", "apply_assistant", True),
            CopyField("求职动机 / 为什么这家", "用下方 motivation 类答案", "apply_assistant", True),
            CopyField("实习时长 / 周到岗", "实习类岗诚实填; 北森筛 3+ 月较多", "用户确认"),
            CopyField("附件 PDF 文件名", _resume_filename_hint(company, title), "tailor_resume", True),
        ],
        material_checklist=[
            *_common_materials(company, title),
            "北森 ATS 简历 PDF 5MB 限制, 字体内嵌避免解析乱码。",
            "招聘 form 多为多页, 中途别关浏览器, 答完才能保存。",
        ],
        post_apply_actions=_common_post_actions(),
        verified_source=verified_source, evidence_url=evidence_url,
    )


def _shixiseng_plan(
    *, url: str | None, company: str, title: str,
    verified_source: bool, evidence_url: str | None,
) -> ApplicationPlan:
    """W20 — 实习僧 实习专用聚合.

    实习僧的投递流程: 站内注册 → 简历填写/上传 → 点"投递" → HR 在站内
    后台收到 → HR 直接发消息或加微信. 不像牛客有"沟通"功能, 但有"打招呼"
    + 简历同步 + 站内信回应. 数据 verified 2026-05-11.
    """
    return ApplicationPlan(
        platform="shixiseng",
        platform_label="实习僧 · 实习专用聚合",
        apply_url=url, action_label="打开实习僧投递",
        channel_note=(
            "实习僧是国内主流大学生实习入口, 站内注册后投递 (微信/手机号都行)。"
            "投递后 HR 在站内后台直接看简历, 看上的会发站内信或加微信; 实习僧"
            "也支持上传 PDF 简历做附件投递。"
        ),
        steps=[
            "微信扫码或手机号注册 shixiseng.com (一次注册可投全站)。",
            "完善在线简历 — 学校/专业/毕业时间/可实习时长一定填全, 实习僧"
            "HR 主要看这几项过滤。",
            "上传 PDF 附件简历 (针对这家微调过的 master_resume 版本)。",
            "打开 JD 详情, 点「立即投递」按钮 (有的岗要先点「沟通」再投)。",
            "投递后 1-7 天 HR 可能站内信回, 可以同时盯邮件 + 微信加好友请求。",
            "投递后立刻在 OfferGuide 标「我投了」。",
        ],
        fields=[
            CopyField("申请岗位", f"{company} · {title}", "JD", True),
            CopyField("学校/专业/毕业时间", "和简历完全一致, HR 主要看这几项", "用户资料"),
            CopyField("可实习时长 / 周到岗天数", "诚实填; 实习僧很多岗要 3+ 月 / 周 4+ 天",
                      "用户确认"),
            CopyField("项目经历摘要", "用下方 QA 中 project 类答案", "apply_assistant", True),
            CopyField("打招呼/留言 (可选)", "粘下方自我介绍话术, 改成 1-2 句", "apply_assistant", True),
            CopyField("附件 PDF 文件名", _resume_filename_hint(company, title),
                      "tailor_resume", True),
        ],
        material_checklist=[
            *_common_materials(company, title),
            "实习僧 PDF 5MB 以内, 中文字体内嵌避免 HR 看简历乱码。",
            "实习僧 HR 多偏向 1-2 周内能到岗的候选人, 时间灵活就主动写出来。",
        ],
        post_apply_actions=[
            "投后留意实习僧站内信 + 邮箱 + 微信加好友请求 (HR 经常加微信谈)。",
            "投后 7-10 天没消息可以在岗位下方点「再次沟通」or 撤回重投。",
            *_common_post_actions(),
        ],
        verified_source=verified_source, evidence_url=evidence_url,
    )


def _agent_search_plan(
    *, url: str | None, company: str, title: str,
    verified_source: bool, evidence_url: str | None,
) -> ApplicationPlan:
    """W19+ — agent_search 找到的岗位. LLM 经 web search 抓回, 不是 verified API.
    用户必须先核验是否真实存在再投, 否则浪费时间."""
    return ApplicationPlan(
        platform="agent_search_external",
        platform_label="agent 搜到的外部岗 · ⚠ 先核验",
        apply_url=url, action_label="打开链接核验",
        channel_note=(
            "这条 JD 来自 OfferGuide agent 自动 web search, **不是 verified API**。"
            "agent 找的岗位有时是过期 / 错误公司名 / 死链, 投前请先打开链接核验是否真存在。"
            "确认无误后按页面具体投递入口 (大概率是公司官网) 走流程, 用下方材料填写。"
        ),
        steps=[
            "**先打开链接** 看页面是否仍在招 + 公司名是否一致 + 截止日期是否过。",
            "如果链接 404 / 公司不对 → 在 OfferGuide 标「已撤回」, agent 会从池子去掉。",
            "如果真在招 → 按页面提示走 (官网注册 / 邮箱投 / BOSS / 等), 用下方 QA 模板填材料。",
            "投递后回 OfferGuide 标「我投了」。",
        ],
        fields=[
            CopyField("公司 / 岗位名", f"{company} · {title}", "JD"),
            CopyField("项目经历摘要", "用下方 QA 中最贴 JD 的 project 答案", "apply_assistant", True),
            CopyField("求职动机", "用下方 motivation 答案", "apply_assistant", True),
            CopyField("附件 PDF 文件名", _resume_filename_hint(company, title), "tailor_resume", True),
            CopyField("投递方式 (邮件 / 表单 / 沟通)", "看页面给的入口, 没说就邮箱直发 hr@", "用户判断"),
        ],
        material_checklist=[
            *_common_materials(company, title),
            "如果公司是 AI 创业公司 (智谱 / 月之暗面 / MiniMax 等), 简历重点突出 LLM/Agent/RL 实战。",
            "邮件投递时主题写 \"应聘 [岗位名] - [姓名] - [学校]\", HR 一目了然。",
        ],
        post_apply_actions=_common_post_actions(),
        verified_source=False,  # agent_search 永远不算 verified
        evidence_url=evidence_url,
    )


def _tencent_fields(company: str, title: str) -> list[CopyField]:
    """Shared field list for tencent (社招用)."""
    return [
        CopyField("申请岗位", f"{company} · {title}", "JD", True),
        CopyField("学校/专业/毕业时间", "和简历一致", "用户资料"),
        CopyField("项目经历", "用下方 QA 中 project 类答案", "apply_assistant", True),
        CopyField("附件 PDF 文件名", _resume_filename_hint(company, title), "tailor_resume", True),
    ]


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
