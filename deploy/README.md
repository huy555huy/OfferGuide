# 部署 OfferGuide

## 启动 web UI（阻塞运行）

```bash
# 一次性 run（重启电脑就停）
python -m offerguide.ui.web
# → http://127.0.0.1:8000
```

## 启动 autonomous daemon（后台 7 个 cron job）

OfferGuide 有 7 个 daemon 每天定时跑：

| Job | 时间 | 干什么 |
|---|---|---|
| `extract_facts` | 02:00 | 抽 SKILL 输出里的事实进 `user_facts` 长期记忆 |
| `discover_jobs` | 06:30 | spider 抓 namewyf/Campus2026 等公司入口 |
| `jd_enrich`     | 06:45 | spider 抓的 thin JD → fetch + LLM 抽 JD 详情 |
| `corpus_classify` | 07:00 | 给 pending 面经跑 quality 分类器 |
| `silence_check` | 09:00 | 沉默 ≥ 7/14/30 天的应用提醒 |
| `corpus_refresh` | Mon 08:00 | DDG 搜面经 + LLM 过滤 + 入库 |
| `brief_update`  | 23:00 | 维护 `company_briefs` 表 |

### 选项 A：launchd（macOS，推荐）

```bash
# 1. 编辑 deploy/launchd/com.offerguide.daemon.plist
#    把 /Users/huy/new_try 替换成你的仓库路径
#    把 /opt/anaconda3/bin/python 替换成 `which python` 输出

# 2. 安装到 LaunchAgents
cp deploy/launchd/com.offerguide.daemon.plist ~/Library/LaunchAgents/

# 3. 加载
launchctl load ~/Library/LaunchAgents/com.offerguide.daemon.plist

# 4. 验证
launchctl list | grep offerguide
tail -f /tmp/offerguide.daemon.log

# 卸载
launchctl unload ~/Library/LaunchAgents/com.offerguide.daemon.plist
```

### 选项 B：tmux + 手跑（开发期）

```bash
tmux new -s offerguide
python -m offerguide.autonomous run
# Ctrl-B D 把 tmux 放后台
```

### 选项 C：systemd（Linux）

写一个 `~/.config/systemd/user/offerguide.service`：

```ini
[Unit]
Description=OfferGuide autonomous daemon

[Service]
WorkingDirectory=/path/to/repo
EnvironmentFile=/path/to/repo/.env
ExecStart=/usr/bin/python -m offerguide.autonomous run
Restart=on-failure
RestartSec=30s

[Install]
WantedBy=default.target
```

然后 `systemctl --user enable --now offerguide`。

## 单次跑某个 job（不启动 daemon）

```bash
python -m offerguide.autonomous run-once discover_jobs
python -m offerguide.autonomous run-once jd_enrich
python -m offerguide.autonomous run-once corpus_classify
```

## 查看 daemon 是否真在跑

打开 web UI → `/dashboard` → 看「daemon 健康」卡片。每个 job 显示：
- 上次运行时间
- 上次结果（counters dict）
- 下次预计运行时间

如果某个 job 显示「从未运行」，daemon 可能没启动 / 该 job 还没到触发时间。
