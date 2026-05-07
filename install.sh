#!/usr/bin/env bash
# OfferGuide 一行装 — 验证环境 + 装 deps + 提示下一步.
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/huy555huy/OfferGuide/main/install.sh | bash
#   # OR after git clone:
#   ./install.sh
#
# 设计原则:
# - 不假设用户有 Python (会装 uv 自动 manage 3.11+)
# - 失败立刻报错 + 给具体修法 (set -euo pipefail)
# - 每步 echo 当前在干啥, 让用户知道为什么慢

set -euo pipefail

# ── colors ─────────────────────────────────────────────────────────
GREEN=$(tput setaf 2 2>/dev/null || true)
RED=$(tput setaf 1 2>/dev/null || true)
YELLOW=$(tput setaf 3 2>/dev/null || true)
DIM=$(tput dim 2>/dev/null || true)
BOLD=$(tput bold 2>/dev/null || true)
RESET=$(tput sgr0 2>/dev/null || true)

ok()   { echo "${GREEN}✓${RESET} $*"; }
warn() { echo "${YELLOW}⚠${RESET} $*"; }
err()  { echo "${RED}✗${RESET} $*" >&2; }
step() { echo ""; echo "${BOLD}▶${RESET} $*"; }
hint() { echo "  ${DIM}$*${RESET}"; }

# ── 1. uv (Python package manager — handles Python install too) ────
step "1/5 检查 uv (Python 包管理器)"
if command -v uv >/dev/null 2>&1; then
    ok "uv $(uv --version | awk '{print $2}') 已装"
else
    warn "uv 没装, 现在装 (用官方 install script)"
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        err "uv 装失败. 手动装: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    # uv 装在 ~/.cargo/bin/uv (Linux/macOS), 当前 shell 可能 PATH 还没有它
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        err "uv 装完了但 PATH 找不到. 手动 source ~/.bashrc 或重启 terminal 再跑 ./install.sh"
        exit 1
    fi
    ok "uv 装好"
fi

# ── 2. project deps ─────────────────────────────────────────────────
step "2/5 装项目依赖 (uv sync)"
if ! uv sync --extra ui --extra autonomous; then
    err "uv sync 失败. 检查 pyproject.toml + 网络. 详细日志在上面"
    exit 1
fi
ok "依赖装好 ($(uv pip list 2>/dev/null | wc -l) 个包)"

# ── 3. .env template ────────────────────────────────────────────────
step "3/5 .env 配置"
if [[ -f .env ]]; then
    ok ".env 已存在 (没动它)"
else
    if [[ -f .env.example ]]; then
        cp .env.example .env
        ok "从 .env.example 复制了 .env"
        hint "下一步: 编辑 .env 填 OFFERGUIDE_LLM_API_KEY (DeepSeek key) + OFFERGUIDE_RESUME_PDF (简历路径)"
    else
        warn ".env.example 不存在 — 你需要手写 .env, 至少:"
        hint 'OFFERGUIDE_LLM_API_KEY="sk-..."'
        hint 'OFFERGUIDE_RESUME_PDF="/path/to/简历.docx"'
    fi
fi

# ── 4. doctor (validate everything) ─────────────────────────────────
step "4/5 体检 (doctor.py 一键诊断)"
if uv run python scripts/doctor.py --quick; then
    ok "体检通过"
else
    warn "体检发现问题 — 上面 ✗ 的每一条都给了具体 fix, 修完再跑 ./install.sh"
    exit 1
fi

# ── 5. instructions ─────────────────────────────────────────────────
step "5/5 装好了"
echo ""
echo "${BOLD}启动:${RESET}"
echo "  ${GREEN}uv run --extra ui python -m offerguide.ui.web${RESET}"
echo ""
echo "${BOLD}然后浏览器打开:${RESET}"
echo "  ${GREEN}http://127.0.0.1:8000${RESET}"
echo ""
echo "${BOLD}首页粘 1 个 JD 链接 / 文本 → 10-20s 出评估报告.${RESET}"
echo ""
hint "完整文档: README.md"
hint "出问题: uv run python scripts/doctor.py --probe-llm"
