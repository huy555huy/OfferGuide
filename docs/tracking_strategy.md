# Application Tracking Strategy

> 实现状态："schema 在，detector 部分有，部分待补"。本文诚实记录每个信号源的当前状态。

OfferGuide 的 `application_events` 表是事件日志（append-only）。每条事件有 `kind` (submitted / viewed / replied / assessment / interview / rejected / offer / withdrawn / silent_check) 和 `source` (manual / email / platform / calendar / inferred)。**问题不是 schema，是 detector**——怎么自动把现实世界的信号转成事件。

## 5 个信号源 + 当前实现状态

| 信号 | 来源 | 实现路径 | 现状 |
|---|---|---|---|
| ① **手动输入** | 用户自己看到/听到 → 在 UI 点按钮 | `/applications` 页 → 1-click 按钮（HR 看了 / 回复了 / 笔试 / 面试 / 拒了 / Offer）→ `POST /api/applications/{id}/event` | ✅ **已实现**（W8' UI 大改） |
| ② **Boss 已查看 / 站内信** | Boss 自己 UI 显示"已查看"badge + 新消息 | 浏览器扩展定期扫 `https://www.zhipin.com/web/geek/recommend` + 站内信列表，解析变化 → POST 到本地 | ⏳ **未实现**（扩展只做 JD 提取） |
| ③ **牛客投递记录** | 牛客 `/v3/process` 页面显示状态变化 | 同 Boss：扩展定期 fetch 该页 + 解析 | ⏳ **未实现** |
| ④ **邮件**（80% 校招通信走这） | 笔试链接 / 面试邀约 / 拒信 / Offer 信 | 用户**应用密码**（不是真密码）+ 本地 IMAP 拉收件箱 + pattern 匹配 | ⏳ **未实现**（隐私敏感，单独 phase） |
| ⑤ **日历 ICS** | 面试邀约通常带 `.ics` 附件 | ICS 解析 → `kind='interview'` + payload 含 round/scheduled_at | ⏳ **未实现**（单文件上传 ~50 行 Python） |

## 信号 ① 已实现（W8'）

UI 长这样（`/applications`）：

```
[applied] [😴沉默 15d]                              ●● 2 事件
推荐算法实习
美团 · nowcoder
▸ 事件时间线 (2)

日志事件: 👀 已查看  💬 已回复  🧪 笔试  🎤 面试  🚫 拒了  🎉 Offer
```

每个按钮 POST 到 `/api/applications/{id}/event` (`kind=viewed|replied|...`)，后端：
1. `application_events.record(...)` 入 append-only 日志
2. `state_machine.sync_status(...)` 更新 `applications.status` 派生列
3. HTMX 把这一行 swap 回来（不刷新整页）

**这一招覆盖了 80% 的实际场景**：
- ✅ 微信 / 电话沟通: 用户看到消息后手点按钮
- ✅ 官网投递（无平台 API）: 同上
- ✅ Boss / 牛客（在自动 detector 落地前）: 同上
- ✅ 内推: 同上

## 信号 ② / ③ 落地路径（W9 候选）

### Boss 浏览器扩展事件抓取

扩展现在的 manifest:

```json
"content_scripts": [{
  "matches": ["*://*.zhipin.com/job_detail/*"],
  "js": ["content.js"]
}]
```

要扩展为：

```json
"content_scripts": [{
  "matches": [
    "*://*.zhipin.com/job_detail/*",        // 现有: JD 提取
    "*://*.zhipin.com/web/geek/recommend*", // 新: 推荐列表 + 已查看
    "*://*.zhipin.com/web/chat*"            // 新: 站内信
  ]
}],
"permissions": ["storage", "alarms"]   // 新: 用 alarms 定期扫描
```

`background.js` 用 `chrome.alarms` 每 10 min 触发一次扫描脚本 → 解析 DOM → diff 上次状态 → 检测出新 viewed/replied 信号 → POST `http://localhost:8000/api/applications/{id}/event`。

**关键工程问题**：
- application_id 怎么对应 Boss 上的某个 JD？需要在初次抓 JD 时记录 Boss 的 `encryptUid` / `jobId` 到 `applications.payload`，扫描时用这个 key 查找
- 用户没登录 / 切换账号 → 扩展应静默退出，不 POST 错事件
- 速率限制 / 反爬：定期扫描频率 ≤ 1 req/min，遵守 robots.txt

### 牛客同款

`/v3/process` 是牛客的"投递进度"页，DOM 结构稳定。同 Boss 模式：扩展定期扫描 + diff + POST。

## 信号 ④ 邮件（隐私敏感，单独 phase）

### 推荐路径：Gmail / 163 应用密码 + 本地 IMAP

```python
# 仅本地处理，不上传任何邮件原文
import imaplib

m = imaplib.IMAP4_SSL("imap.gmail.com")
m.login(user, app_password)  # ← 应用密码，不是真密码
m.select("INBOX")
_, ids = m.search(None, '(SINCE "01-Jan-2026")')
for i in ids[0].split():
    _, data = m.fetch(i, "(RFC822)")
    msg = email.message_from_bytes(data[0][1])
    subj = msg["Subject"]
    body = _extract_text(msg)
    detect = _classify(subj, body)  # 见下
    if detect:
        application_events.record(store, application_id=detect.app_id,
                                   kind=detect.kind, source="email",
                                   payload={"subject": subj[:200]})
```

### 分类器（轻量 pattern + 可选 LLM）

第一层 keyword pattern（覆盖 90%）：

```python
PATTERNS = {
    "interview":  [r"面试.*邀请", r"interview.*invitation", r"面试通知"],
    "assessment": [r"笔试.*链接", r"在线测评", r"OA.*test"],
    "rejected":   [r"很遗憾", r"未能", r"不合适", r"thank you for.*application"],
    "offer":      [r"offer", r"录用通知", r"非常荣幸"],
    "replied":    [r"回复.*您", r"已收到.*简历"],
}
```

第二层（pattern 没命中但邮件来自已知公司域名）→ 调一次 LLM 分类。

### 隐私 / 安全规则

1. **Application Password ≠ 真密码**：让用户在 Gmail/163 设置里生成专用密码
2. **本地处理**：永远不上传邮件原文到 LLM 服务商
3. **明确选择性**：用户在 settings 里勾选信任的发件人域名（@bytedance.com / @alibaba-inc.com / ...）
4. **opt-in only**：默认关闭，UI 提示用户需要主动开

实现成本估算：~300 行 Python + 1 周打磨 + 安全 review。这是 **W9 后** 的工作。

## 信号 ⑤ 日历 ICS（小工程量）

面试邀约邮件**通常带 `.ics` 附件**。用户可以：
1. 在 web UI 上传 .ics 文件
2. 或者邮件 detector（信号 ④）自动 detect 附件 + 解析

```python
import icalendar

cal = icalendar.Calendar.from_ical(open(ics_path).read())
for component in cal.walk():
    if component.name == "VEVENT":
        summary  = component.get("summary")
        start_dt = component.get("dtstart").dt
        # heuristic: summary contains 面试 / interview → record event
        if any(k in str(summary) for k in ("面试", "interview")):
            application_events.record(
                store, application_id=app_id,
                kind="interview", source="calendar",
                occurred_at=_to_julian(start_dt),
                payload={"summary": str(summary), "scheduled_at": start_dt.isoformat()}
            )
```

**实现成本**：~50 行 Python + UI 表单一个文件上传 input。**值得做**——一个 .ics 上传就能写 `interview` + `payload.scheduled_at`，准确率高。

## 官网投递怎么追踪

校招很多走公司官网（zhaopin.alibaba.com、jobs.bytedance.com、careers.tencent.com 等），不走 Boss/牛客。**不暴露"已查看"信号给求职者看**，所以只能靠：

1. **采集 JD**：浏览器扩展加新 `content_scripts.matches` + 写每个站的 parser（手工活，每家页面结构不同）
2. **追踪状态**：依赖**邮件**（信号 ④）+ **手动 log**（信号 ①）
3. **应用源头标记**：jobs.source = `'official_site'`（W8' 已添加，UI 区分显示）

## 总结

| 阶段 | 涵盖比例 | 当前 |
|---|---|---|
| **手动 1-click**（信号 ①） | ~50%（最关键 baseline） | ✅ W8' done |
| **+ Boss/牛客 自动**（②③） | ~75% | ⏳ W9 候选 |
| **+ 邮件 IMAP**（④） | ~95% | ⏳ W10+ |
| **+ ICS**（⑤） | ~95%（覆盖度高的最后 5%） | ⏳ small task |

**优先级**：① 已 done > ⑤（最便宜）> ② / ③（中等工程量）> ④（最重，隐私敏感）。
