# OfferGuide

OfferGuide 是一个本地优先的中文求职 Agent。产品只围绕三件事展开：找到值得投的岗位、为当前 JD 准备一份可直接使用的投递包、在真实投递后查找真实面经并回答原题。

```text
多源找岗 → 用户选择 → JD 定制简历 + 投递材料 → 用户提交
                                              ↓
                             实际提交版本 → 真实面经原题与答案
```

用户手动粘贴 JD 是兜底入口。OfferGuide 不替用户点击发送或提交。

## 当前主流程

### 1. 找岗位

- `JobDiscoveryAgent` 根据当前求职意图自主选择招聘平台、官方页面和网页搜索工具。
- 只有取得可追溯来源和完整 JD 的岗位才能进入当前选择集；Agent 同时给出推荐理由、顾虑和未知项。
- `/recommended` 是候选岗位视图；用户从这里选择要推进的岗位。

### 2. 做投递包

- 读取用户主简历和当前 JD。
- 由模型根据当前 JD 决定内容取舍、排序、展开程度和表达。
- 使用唯一的 Typst 模板生成一份 PDF；模板以用户确认的参考简历为视觉基准。
- 同时准备当前渠道需要的投递话术、网申回答和提交前检查。
- 用户打开 PDF 检查后自行投递；点击“我投了”只记录并冻结实际提交版本。

### 3. 查面经并回答原题

- 使用被冻结的 JD 和简历，而不是之后变化的主简历。
- `InterviewResearchAgent` 按工作内容相似度搜索公开、可读取的真实面经；同公司是加分项，不是硬条件。
- 只有原帖中实际出现的问题才能进入结果，答案结合被冻结的 JD、对方收到的简历和 Project Vault。
- 没有找到可读取的真实相似岗位面经时显示 `not_found`，不生成预测题或替代材料。

## Quick Start

推荐使用 `uv`：

```bash
git clone https://github.com/huy555huy/OfferGuide.git
cd OfferGuide
uv sync --extra dev --extra ui
cp .env.example .env
```

编辑 `.env`：

```bash
OFFERGUIDE_LLM_API_KEY="sk-..."
OFFERGUIDE_RESUME_PDF="/absolute/path/to/resume.pdf"
TAVILY_API_KEY="tvly-..."  # 自动找岗和公开面经搜索

# 可选
OFFERGUIDE_LLM_BASE_URL="https://api.deepseek.com"
OFFERGUIDE_LLM_MODEL="your-model"
```

启动：

```bash
uv run python -m offerguide.ui.web
```

默认地址：`http://127.0.0.1:8000`

关闭 Web 进程内的后台 Research Agent 刷新：

```bash
OFFERGUIDE_NO_BACKGROUND_AGENTS=1 uv run python -m offerguide.ui.web
```

## 关键模块

```text
src/offerguide/
  research_agents/
    runner.py             # 模型自主工具循环与领域完成状态
    sources.py            # 搜索、原文保存、分页读取与来源状态
    service.py            # Web、后台刷新和主 Agent 共用的唯一装配
    job_discovery/        # 找岗上下文、岗位证据与当前选择集
  interview_research/     # 公开原帖、真实问题与当前答案集
  platforms/              # JobDiscoveryAgent 使用的真实平台适配器
  resume/models.py        # 唯一 ResumeDocument 与完整 Resume Context
  resume/editor.py        # 内容编辑、用户反馈和页面视觉审阅
  resume/render.py        # 确定性 Typst PDF 与逐页 PNG
  resume/workspace.py     # 每个申请唯一 draft 与 submitted 冻结
  resume/workflow.py      # Context → 编辑 → 渲染 → 投递包主链
  skills/apply_assistant/ # 当前投递话术、网申回答和提交前检查
  ui/                     # FastAPI + Jinja 页面
```

主对话 Agent 和诊断记录是支撑能力；它们不能绕过两个领域 Agent 另写岗位选择集或面试材料，也不是产品完成度的替代品。

## 简历原则

- 不编造学校、公司、时间、项目、奖项、业绩或数字结果。
- 可以根据 JD 重新选择、排序和表达已有经历。
- 可以加入与已有能力相邻、可在面试前准备到可解释程度的知识或工具声明，但不能伪装成已经发生的项目成果。
- 篇幅随当前岗位需要和真实证据自然展开，模板不为适配预设版面压缩内容。
- 最终 PDF 的实际可读性高于内部规则或测试指标。

## 测试

```bash
uv run pytest -q
git diff --check
```

测试用于防回归；页面和 PDF 仍需打开检查实际效果。

## 配置

| 变量 | 说明 |
|---|---|
| `OFFERGUIDE_LLM_API_KEY` | OpenAI-compatible API key |
| `OFFERGUIDE_LLM_BASE_URL` | API base URL |
| `OFFERGUIDE_LLM_MODEL` | 模型名 |
| `OFFERGUIDE_RESUME_PDF` | 主简历 PDF 路径；只提取文本，最终格式由统一模板生成 |
| `TAVILY_API_KEY` | 自动找岗和公开面经 Search/Extract；没有时仍可处理用户主动粘贴的正文 |
| `OFFERGUIDE_VISION_API_KEY` | 可选，多模态页面审阅模型的 API key |
| `OFFERGUIDE_VISION_BASE_URL` | 可选，多模态 OpenAI-compatible base URL |
| `OFFERGUIDE_VISION_MODEL` | 可选，必须能接收 `image_url` 的模型名 |
| `OFFERGUIDE_DB` | SQLite 路径 |
| `OFFERGUIDE_PORT` | Web 端口 |
| `OFFERGUIDE_NO_BACKGROUND_AGENTS` | `1` 时关闭后台 Research Agent 刷新 |

## License

MIT. See [LICENSE](LICENSE).
