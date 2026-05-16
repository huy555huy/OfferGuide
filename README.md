# OfferGuide

OfferGuide 是一个面向中文求职场景的本地优先 agent。它围绕用户的目标、简历、岗位、投递状态和反馈持续维护自己的状态，并在每次 wake 时决定下一步是行动、追问、通知还是等待。

项目当前的重点是让 agent 拥有可恢复的世界状态:

- `worldview`: 用户画像、策略、开放回路、复盘和长期记忆
- `agent_work_items`: agent 正在处理、等待、阻塞或已停止关注的工作项
- `harness_runs`: 每次运行的触发、工具调用、状态和成本记录
- `harness_events`: 投递、评分、面试、反馈等事实事件
- `scheduled_wakes`: agent 自己留下的未来检查点

UI 是这些状态的观察窗口。用户可以从页面查看和修正状态，但主路径应从 Agent Chat 开始。

## 当前入口

| 入口 | 路径 | 用途 |
|---|---|---|
| Mission Control | `/` | agent 当前状态、开放工作、输入框和近期活动 |
| Pipeline | `/pipeline` | 候选岗位、投递状态、应用时间线和转化概览 |
| Tailor | `/tailor` | 简历上下文、定向修改建议和项目事实引用 |
| Interviews | `/interviews` | 面试准备、面经、复盘上下文 |
| Agent Runs | `/agent/runs/{id}` | 单次 agent 运行轨迹和工具调用 |
| Lab / Debug | `/evolution`, `/debug`, `/metrics` | SKILL 演化、调试和系统观测 |

部分旧路由仍保留兼容，例如 `/recommended`, `/applications`, `/funnel`, `/jobs`, `/compare`, `/mock`, `/reflect`。它们会逐步收敛为 Pipeline / Tailor / Interviews 下的状态切面或详情页。

## Quick Start

推荐用 `uv`:

```bash
git clone https://github.com/huy555huy/OfferGuide.git
cd OfferGuide

uv sync --extra dev --extra ui --extra evolution --extra scheduling
cp .env.example .env
```

编辑 `.env`:

```bash
OFFERGUIDE_LLM_API_KEY="sk-..."
OFFERGUIDE_RESUME_PDF="/path/to/your/resume.docx"

# 可选: web search / proactive discovery
TAVILY_API_KEY="tvly-..."

# 可选: OpenAI-compatible endpoint / model
OFFERGUIDE_LLM_BASE_URL="https://api.deepseek.com"
OFFERGUIDE_LLM_MODEL="deepseek-chat"
```

启动 Web UI:

```bash
uv run python -m offerguide.ui.web
```

默认地址:

```text
http://127.0.0.1:8000
```

如果端口被占用:

```bash
OFFERGUIDE_PORT=8770 uv run python -m offerguide.ui.web
```

不启用后台 scheduler / ambient crawl:

```bash
OFFERGUIDE_NO_SCHEDULER=1 uv run python -m offerguide.ui.web
```

也可以用 pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui,evolution,scheduling]"
python -m offerguide.ui.web
```

## Agent Runtime

主运行时在 [`src/offerguide/agent_runtime`](src/offerguide/agent_runtime):

| 文件 | 作用 |
|---|---|
| `loop.py` | 单线程 agent loop: build context -> LLM tool call -> dispatch -> continue / stop |
| `instructions.md` | agent 的运行契约，包含 Observe / Agenda / Decide / Act / Verify / Sleep |
| `context.py` | 每次 wake 注入系统事实、active goals、work items、agenda、worldview memory |
| `_schema.py` | 运行时表: runs, events, scheduled wakes, work items |
| `tools.py` | main-agent 工具 schema 和 dispatch |
| `memory.py` | `.offerguide/worldview/*.md` 的读写工具 |
| `triggers.py` | user input / event / scheduled / cron 触发 |

每次运行的大致过程:

```text
trigger
  -> materialize work item
  -> build system context
  -> model chooses tools
  -> tools update DB / worldview / inbox / wakes
  -> agent updates work item state
  -> run record is persisted
```

`agent_work_items` 是 agent 自己的注意力账本。状态包括:

```text
open -> in_progress -> done
                      -> waiting
                      -> blocked
                      -> dismissed
```

完成、等待、阻塞和丢弃都必须写回 durable state。下一次 wake 只应看到仍需继续处理的工作。

## Worldview

默认目录:

```text
.offerguide/worldview/
```

关键文件:

| 文件 | 内容 |
|---|---|
| `MEMORY.md` | agent 的主页和跨 wake 摘要 |
| `agenda.md` | 开放回路、阻塞、机会和安静等待项 |
| `candidate.md` | 用户画像、简历事实、偏好和边界 |
| `tracked-jobs.md` | 关注中的岗位和判断 |
| `upcoming-events.md` | 面试、deadline、follow-up |
| `reflections.md` | agent 对自己行为和用户反馈的复盘 |
| `strategy.md` | 当前求职策略和项目策略 |

这些文件会进入 agent 的上下文，影响下一次 wake 的判断。

## Core Capabilities

OfferGuide 当前包含以下能力:

- JD ingestion: 用户粘 URL 或 JD 文本后进入 jobs 表
- Match scoring: 基于候选人事实和 JD 事实评估匹配
- Resume tailoring: 生成有证据边界的定向修改建议
- Project Vault: 保存项目事实、可讲边界和不可声称内容
- Interview prep: 面试上下文、准备重点和复盘记录
- Pipeline tracking: considered / applied / interview / rejected / offer 等状态
- Scheduled wakes: follow-up、deadline、等待事项的未来检查点
- Skill evolution: 基于用户反馈、应用结果和 follow-through 信号维护 SKILL variants
- Browser extension: 从 BOSS 页面提取 JD / 推荐列表并同步到本地服务

## Chrome Extension

加载方式:

```text
Chrome -> chrome://extensions -> Developer mode -> Load unpacked -> browser_extension/
```

后端需要先运行:

```bash
uv run python -m offerguide.ui.web
```

扩展主要负责把页面上的 JD 或推荐列表交给本地 OfferGuide 服务。发送、投递和最终决定仍由用户确认。

## Repository Layout

```text
src/offerguide/
  agent_runtime/      # 主 agent runtime
  agents/             # 子 agent 基类、discovery、evaluation
  skills/             # SKILL.md 单元和 SkillRuntime
  evolution/          # SKILL variant、fitness、release cycle
  tools/              # 工具注册和 discovery/evaluation 工具
  workers/            # scout / tracker / ambient workers
  autonomous/         # scheduler 和后台 job
  platforms/          # nowcoder / official_jobs / shixiseng / zerovoice 等来源
  memory/             # SQLite store 和 sqlite-vec
  profile/            # 简历读取
  ui/                 # FastAPI + Jinja UI

browser_extension/    # Chrome extension
docs/                 # 设计记录、审计、dogfood 和研究材料
tests/                # pytest 测试
PROGRESS.md           # 当前进度和 handoff
test-results.json     # 长任务验证状态
```

## Development

运行测试:

```bash
uv run pytest -q
```

运行 runtime 相关测试:

```bash
uv run pytest \
  tests/test_w15_harness.py::TestToolDispatch \
  tests/test_w15_harness.py::TestHarnessLoop \
  tests/test_w15_harness.py::TestTriggers \
  tests/test_w15_harness.py::TestContextManager \
  tests/test_evidence_first_context.py \
  -q
```

检查格式类问题:

```bash
git diff --check
```

查看当前项目目标和过程记录:

```text
docs/goal_real_agent_2026-05-16.md
docs/feature_necessity_audit_2026-05-16.md
PROGRESS.md
```

## Configuration

常用环境变量:

| 变量 | 说明 |
|---|---|
| `OFFERGUIDE_LLM_API_KEY` | OpenAI-compatible LLM API key |
| `OFFERGUIDE_LLM_BASE_URL` | LLM API base URL |
| `OFFERGUIDE_LLM_MODEL` | 默认模型 |
| `OFFERGUIDE_RESUME_PDF` | 用户简历路径，支持 pdf / docx |
| `TAVILY_API_KEY` | web search / discovery |
| `OFFERGUIDE_DB` | SQLite DB 路径，默认 `.offerguide/store.db` |
| `OFFERGUIDE_PORT` | Web 端口，默认 `8000` |
| `OFFERGUIDE_NO_SCHEDULER` | 设为 `1` 时关闭 in-process scheduler |
| `OFFERGUIDE_NOTIFY` | `console`, `feishu`, `telegram` |

## Current Status

当前主线已经完成:

- redesign shell 接入核心页面
- Pipeline / Tailor / Interviews 的 IA 初步收敛
- active goals / agenda / worldview 注入 agent context
- `agent_work_items` durable state
- `update_work_item` 工具，支持 done / waiting / blocked / dismissed
- feature necessity audit 和 real-agent 过程记录

最近一次完整验证:

```text
866 passed, 2 skipped
```

下一步优先级:

1. 用真实 JD 做端到端 dogfood: user input -> work item -> tools -> state reference -> work item status update
2. 继续收敛旧的一等入口，把它们并入 Pipeline / Tailor / Interviews 的状态切面
3. 补齐用户 candidate facts，再扩大主动 discovery

## License

MIT. See [LICENSE](LICENSE).
