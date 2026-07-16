# 部署 OfferGuide

## 单一服务进程

OfferGuide 不再运行独立 autonomous daemon。Web、定期找岗刷新、提交后的面经研究和主 Agent 委托都通过同一个 `ResearchAgentService` 使用 `JobDiscoveryAgent` 与 `InterviewResearchAgent`；额外启动第二套 scheduler 会破坏唯一结果所有权。

安装并启动：

```bash
uv sync --extra ui
uv run python -m offerguide.ui.web
```

默认地址为 `http://127.0.0.1:8000`。进程停止后页面与后台研究任务都会停止；已经写入 SQLite 的岗位证据、冻结投递和当前材料不会丢失。

## 后台 Research Agent

默认情况下，Web lifespan 会定期触发同一个 `JobDiscoveryAgent`；实际提交、面试上下文更新和用户刷新会异步触发绑定当前 submitted workspace 的 `InterviewResearchAgent`。触发器只负责唤醒，不拥有搜索、排序或材料生成逻辑。

开发或诊断时可以关闭后台刷新：

```bash
OFFERGUIDE_NO_BACKGROUND_AGENTS=1 uv run python -m offerguide.ui.web
```

这个开关不创建替代 daemon；用户仍可在页面显式请求找岗或面经研究。

## 长期运行

需要 tmux、launchd 或 systemd 托管时，进程管理器只启动同一条 Web 命令。例如开发机可以使用：

```bash
tmux new -s offerguide
uv run python -m offerguide.ui.web
# Ctrl-B D 把 tmux 放后台
```

systemd 示例：

```ini
[Unit]
Description=OfferGuide web and research agents

[Service]
WorkingDirectory=/path/to/repo
EnvironmentFile=/path/to/repo/.env
ExecStart=/path/to/repo/.venv/bin/python -m offerguide.ui.web
Restart=on-failure
RestartSec=30s

[Install]
WantedBy=default.target
```

不要再部署 `offerguide.autonomous`、`wake_agent` 或旧 corpus/score jobs；这些入口已退出生产。

## 验证

打开 `/recommended` 检查当前找岗状态，完成一次实际提交后打开对应投后页面检查面经研究状态。运行记录可以在 Web 调试页查看；真实岗位来源、完整 JD、冻结投递和可打开的面经引用才是链路是否可用的依据。
