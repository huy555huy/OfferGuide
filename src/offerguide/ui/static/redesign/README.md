# OfferGuide · Web Redesign

融合方向 · 7 个 hi-fi mockup · 1440×900

## 看法

直接打开 **`index.html`** — 顶栏切换 50/65/80/100% 缩放，章节跳转 memo / 每日 / 单 JD 流 / Lab。每个 frame 标题右上 *独立打开 ↗* 可以全屏看单屏。

## 文件结构

```
index.html              ─ 装载页（7 屏 inline 在 srcdoc，离线可看）
tokens/a.css            ─ 设计 tokens（颜色 / 字体 / bracket-tag / uptime / kbar）
screens/
  a-home.html           ─ 01a · Mission Control
  a-pipeline.html       ─ 01b · Pipeline 看板
  a-drawer.html         ─ 02a · Decision Drawer（单 JD 评估抽屉）
  a-tailor.html         ─ 02b · Tailor（简历微调）
  a-interview.html      ─ 02c · Interview Prep
  a-lab.html            ─ 03a · Lab · SKILL 自进化
  a-extension.html      ─ 03b · Chrome 扩展浮窗
```

## 设计语言摘要

| 维度 | 选择 | 备注 |
|---|---|---|
| 主背景 | `#EFEDE5` bone | 冷调暖纸 · 不是 Claude warm cream |
| 文字 | `#0F1419` cool slate + 三档 ink | 中文为主语，冷调 |
| 信号色 | `#16365C` editorial deep navy | Penguin-classic / 编辑部书籍 |
| 状态 | `#1F5C42` 深森林 / `#8C5E14` 深琥珀 / `#8A2A24` 酒红 | 都用深沉版本，不抢信号色 |
| 显示字体 | Noto Serif SC | 大标题 + 关键数字（serif 情绪） |
| 正文字体 | Inter + PingFang SC | 中英混排 |
| 等宽字体 | JetBrains Mono | id / 数据 / 命令 |
| 状态 | uptime / wake / budget 在顶栏 | 像后台进程 |
| 决策 | **INTERRUPT** 中断块 | 不是"提醒"，是 agent 停下来等你 |
| 标签 | `[bracket_tag]` mono | `[ask_user]` / `[scored]` / `[canary]` |

色卡灵感：Penguin Classics + 老式工程手册 + 编辑部书籍设计。在中文语境里更"严肃 + 有质感"，跟项目的 long-running harness / GEPA closed-loop 那套硬核感对得上。

## 信息架构

```
旧 19 page                                          → 新结构
/today /agent /inbox /goals /dashboard              → Mission Control
/recommended /jobs /pipeline /applications /funnel  → Pipeline (1 屏 = 1 条线)
/tailor /project-vault                              → Tailor
/interviews /mock /reflect /stories                 → Interview
/evolution /metrics /portfolio /debug               → Lab
/compare · 单 JD 评估弹窗 · BOSS 扩展                 → 右侧抽屉 + 扩展浮窗
```

## 改代码时怎么接

每个 screen 文件已经是一个独立可跑的 HTML，可以直接：

1. 把 `tokens/a.css` 作为全局 CSS（替换或并入现有 `base.html` 的 `<style>` 块）
2. 把每个 screen 的 `<body>` 部分作为 Jinja `{% block content %}` 接进现有 FastAPI + HTMX 框架
3. 19 个旧 page 路由按上面的 IA 表收敛到 5 个 + 1 个抽屉
4. agent live rail (右侧 340px) 抽成一个 partial template，所有页面 include

## 还能继续做

- 暗黑模式（paper 反相 charcoal，一套 token 双主题）
- 移动端窄屏视图（inbox 刷起来）
- 把 Lab 那屏抽出来做 2 分钟讲项目的 portfolio piece
- 接进代码后的真数据校准（match dimension 字段、prep_plan 模板）
