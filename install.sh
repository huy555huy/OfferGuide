#!/usr/bin/env bash
# OfferGuide 装机 — W15.18 重写, 防冲掉用户已有 working setup.
#
# Usage:
#   ./install.sh                  # 自动检测 + 装
#   ./install.sh --force-uv       # 强制走 uv 路径 (即使已有 conda venv)
#   ./install.sh --skip-deps      # 跳过 uv sync / pip install (假设已装)
#
# 设计: 先检测当前 setup, 不撞坑, 不替换用户已有 .venv.
#   - 已有 .venv 能 import offerguide → 直接 skip 装包
#   - 已在 conda env (CONDA_DEFAULT_ENV 非 base) → 用 pip install -e 不用 uv
#   - 真的什么都没有 → 装 uv 走 fresh path

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

# ── parse args ─────────────────────────────────────────────────────
FORCE_UV=0
SKIP_DEPS=0
for arg in "$@"; do
    case "$arg" in
        --force-uv) FORCE_UV=1 ;;
        --skip-deps) SKIP_DEPS=1 ;;
        -h|--help)
            echo "用法: ./install.sh [--force-uv] [--skip-deps]"
            echo ""
            echo "  默认行为: 检测当前 Python setup, 用最不冲突的路径装包"
            echo "  --force-uv: 即使已有 conda env, 也强装 uv venv (会创建 .venv 子目录)"
            echo "  --skip-deps: 跳过装包 (假设已装), 只跑 doctor + 提示启动"
            exit 0
            ;;
        *) err "未知参数: $arg"; exit 1 ;;
    esac
done

# ── 1. 检测当前 Python setup ───────────────────────────────────────
step "1/4 检测你当前的 Python setup"

PY=""
PIP=""
DEPS_INSTALLED=0
INSTALL_MODE=""  # "conda" | "venv" | "uv" | "fresh"

# 1.1 已经在 conda env 里? (不是 base)
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]] && [[ "${CONDA_DEFAULT_ENV}" != "base" ]]; then
    ok "在 conda env: ${CONDA_DEFAULT_ENV}"
    PY="python"
    PIP="pip"
    INSTALL_MODE="conda"

    # 测一下能不能 import offerguide
    if python -c "import offerguide" 2>/dev/null; then
        ok "offerguide 已 import (deps 都齐了)"
        DEPS_INSTALLED=1
    fi

# 1.2 已经在 .venv 里 (VIRTUAL_ENV 是 repo 下的 .venv)?
elif [[ -n "${VIRTUAL_ENV:-}" ]] && [[ "$VIRTUAL_ENV" == */new_try/.venv* ]]; then
    ok "在 .venv: $VIRTUAL_ENV"
    PY="python"
    PIP="pip"
    INSTALL_MODE="venv"
    if python -c "import offerguide" 2>/dev/null; then
        ok "offerguide 已 import"
        DEPS_INSTALLED=1
    fi

# 1.3 项目根有 .venv 但没 activate?
elif [[ -d ".venv" ]] && [[ -x ".venv/bin/python" ]]; then
    ok ".venv 存在但没 activate"
    if .venv/bin/python -c "import offerguide" 2>/dev/null; then
        ok ".venv 里 offerguide 已 import"
        DEPS_INSTALLED=1
        INSTALL_MODE="venv"
        PY=".venv/bin/python"
        PIP=".venv/bin/pip"
        hint "提示: 跑前先 'source .venv/bin/activate'"
    else
        warn ".venv 里没装 offerguide"
        PY=".venv/bin/python"
        PIP=".venv/bin/pip"
        INSTALL_MODE="venv"
    fi

# 1.4 真的什么都没有 → uv fresh path
else
    if [[ $FORCE_UV -eq 1 ]]; then
        warn "--force-uv: 走 fresh uv path (会创建/重建 .venv)"
    else
        warn "没检测到 conda env / .venv — 走 fresh uv path"
    fi
    INSTALL_MODE="uv"
fi

# 1.5 强制 uv 模式 — 用户显式要求
if [[ $FORCE_UV -eq 1 ]] && [[ "$INSTALL_MODE" != "uv" ]]; then
    warn "你已经在 $INSTALL_MODE 里, 但 --force-uv 强制走 uv. 这会创建一个 fresh .venv 替代当前."
    read -p "  确定要这么干吗? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        warn "已取消. 移除 --force-uv 重新跑"
        exit 0
    fi
    INSTALL_MODE="uv"
fi

# ── 2. 装 deps (按 mode 走最少摩擦的路径) ──────────────────────────
step "2/4 装项目依赖"

if [[ $SKIP_DEPS -eq 1 ]]; then
    ok "跳过 deps 装 (--skip-deps)"
elif [[ $DEPS_INSTALLED -eq 1 ]]; then
    ok "deps 已装 — skip (不冲已有的)"
else
    case "$INSTALL_MODE" in
        conda|venv)
            warn "用 $INSTALL_MODE 模式 — pip install -e \".[ui,autonomous]\""
            if ! $PIP install -e ".[ui,autonomous]"; then
                err "pip install 失败. 看上面错日志, 或试 --force-uv 走 uv 路径"
                exit 1
            fi
            ok "deps 装好"
            ;;
        uv)
            # uv check
            if ! command -v uv >/dev/null 2>&1; then
                warn "uv 没装, 现在装 (官方 install script)"
                if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
                    err "uv 装失败. 手动装: https://docs.astral.sh/uv/getting-started/installation/"
                    exit 1
                fi
                export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
                if ! command -v uv >/dev/null 2>&1; then
                    err "uv 装完了但 PATH 找不到. 重启 terminal 再跑 ./install.sh"
                    exit 1
                fi
            fi
            ok "uv $(uv --version | awk '{print $2}')"

            if ! uv sync --extra ui --extra autonomous; then
                err "uv sync 失败. 详见上面错日志"
                exit 1
            fi
            ok "deps 装好 (uv 模式)"
            PY="uv run python"
            ;;
    esac
fi

# ── 3. .env 配置 ──────────────────────────────────────────────────
step "3/4 .env 配置"
if [[ -f .env ]]; then
    ok ".env 已存在 — 没动它"
else
    if [[ -f .env.example ]]; then
        cp .env.example .env
        ok "从 .env.example 复制了 .env"
        hint "下一步: 编辑 .env 填:"
        hint "  OFFERGUIDE_LLM_API_KEY=\"sk-...\"  (DeepSeek 等 OpenAI compat key)"
        hint "  OFFERGUIDE_RESUME_PDF=\"/path/to/简历.docx\""
    else
        warn ".env.example 不存在 — 你需要手写 .env, 至少:"
        hint 'OFFERGUIDE_LLM_API_KEY="sk-..."'
        hint 'OFFERGUIDE_RESUME_PDF="/path/to/简历.docx"'
    fi
fi

# ── 4. doctor (validate) ──────────────────────────────────────────
step "4/4 体检 (doctor.py 一键诊断)"
DOCTOR_CMD="$PY scripts/doctor.py --quick"
if [[ "$INSTALL_MODE" == "uv" ]]; then
    DOCTOR_CMD="uv run python scripts/doctor.py --quick"
fi

if eval "$DOCTOR_CMD"; then
    ok "体检通过"
else
    warn "体检发现问题 — 上面 ✗ 的每一条都给了具体 fix, 修完再跑"
    exit 1
fi

# ── 5. 启动提示 ───────────────────────────────────────────────────
echo ""
echo "${GREEN}${BOLD}全部就绪.${RESET}"
echo ""
echo "${BOLD}启动:${RESET}"
case "$INSTALL_MODE" in
    conda|venv)
        if [[ -n "${VIRTUAL_ENV:-}" ]] || [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
            echo "  ${GREEN}python -m offerguide.ui.web${RESET}"
        else
            echo "  ${GREEN}source .venv/bin/activate && python -m offerguide.ui.web${RESET}"
        fi
        ;;
    uv)
        echo "  ${GREEN}uv run --extra ui python -m offerguide.ui.web${RESET}"
        ;;
esac
echo ""
echo "${BOLD}然后浏览器打开:${RESET} ${GREEN}http://127.0.0.1:8000${RESET}"
echo ""
echo "${BOLD}首次使用:${RESET}"
echo "  • 顶部 hero 粘 1 个 JD → 10-20s 出评估报告"
echo "  • 装 BOSS 浏览器扩展 (browser_extension/) → 在 BOSS 推荐页一键 sync"
echo ""
hint "出问题: 跑 doctor + LLM 检查: $DOCTOR_CMD --probe-llm"
