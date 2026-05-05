"""Runtime configuration — env-var driven, single source of truth.

Why env vars rather than YAML for W4: nothing here is structurally complex
enough to deserve a config file. Notifier choice is one enum, API keys are
opaque strings, paths are paths. YAML can land in W7 if ambient scheduling
needs more knobs.

Settings.from_env() is called once at app/CLI startup; every component takes
the resulting Settings object explicitly so tests can pass synthetic configs
without monkey-patching the environment.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

NotifyChannel = Literal["console", "feishu", "telegram"]

log = logging.getLogger(__name__)


def _load_dotenv_if_present(path: Path = Path(".env")) -> int:
    """Tiny zero-dep loader for a project-local ``.env`` file.

    Why not python-dotenv: keeping the runtime dep tree lean is an explicit
    project value (see pyproject.toml comments). Our `.env` is a flat
    KEY=value file with optional comments; that's ~15 lines to parse.

    Behavior:
      - Looks for ``.env`` relative to the current working directory
        (matches what users expect when they run ``python -m offerguide.ui.web``
        from the project root)
      - Does NOT override existing env vars — anything already set in the
        shell wins, so CLI overrides like ``OFFERGUIDE_DB=... python -m ...``
        keep working
      - Strips matched surrounding quotes (single or double) from values
      - Skips blank lines and ``#`` comments
      - Skipped entirely when ``OFFERGUIDE_SKIP_DOTENV=1`` is set — used by
        the test suite (see ``tests/conftest.py``) so unit tests never
        read the developer's real ``.env`` and accidentally exfiltrate
        API keys into pytest output / CI logs.

    Returns the count of vars actually injected (0 when file missing,
    skipped, or every var was already set externally). Tests use the
    count to assert the loader ran without depending on the contents.
    """
    if os.environ.get("OFFERGUIDE_SKIP_DOTENV") == "1":
        return 0
    if not path.exists():
        return 0
    injected = 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        log.debug("could not read %s: %s", path, e)
        return 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip a single matched pair of quotes (not an arbitrary leading
        # or trailing quote — that would break values like sk-..."weird).
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            injected += 1
    return injected


@dataclass(frozen=True)
class Settings:
    """All runtime configuration. Frozen → safe to share across threads/coroutines."""

    # LLM — name kept ``deepseek_*`` for backwards-compat with existing
    # callers, but in practice these accept any OpenAI-compatible endpoint
    # (DeepSeek native, ccvibe Claude proxy, OpenRouter, one-api, etc.).
    # See ``Settings.from_env`` for the env-var fallback chain.
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    default_model: str = "claude-sonnet-4-6"

    # Notification — at most one of feishu/telegram is used per send,
    # picked from `notify_channel` (or per-call override).
    feishu_webhook_url: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    notify_channel: NotifyChannel = "console"

    # Storage
    db_path: Path = Path(".offerguide/store.db")
    resume_pdf: Path | None = None

    # Web UI
    web_host: str = "127.0.0.1"
    web_port: int = 8000

    @classmethod
    def from_env(cls) -> Settings:
        """Build a Settings from environment variables. Missing → defaults.

        W14.10: a ``.env`` file in the current working directory is
        auto-loaded before reading env vars. Previously the README told
        users to put their LLM key in ``.env`` but nothing actually loaded
        it, so ``python -m offerguide.ui.web`` showed "LLM 未配置" even
        when the user had configured everything correctly. Existing shell
        env vars still win (CLI overrides preserved).

        LLM credentials accept three naming schemes for compatibility:

        - ``DEEPSEEK_API_KEY`` / ``DEEPSEEK_BASE_URL`` — OfferGuide's
          original env names, kept for backwards compatibility
        - ``TOKEN`` / ``BASE_URL`` — short user-friendly names that
          read naturally in a hand-edited ``.env``
        - ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` — fallback for
          OpenAI-compatible proxy deployments

        First-defined wins. Smart-quote-style trailing ``”`` characters
        are stripped (a common copy-paste hazard in Chinese keyboard
        layouts).
        """
        # W14.10: load .env from cwd (idempotent, no-op when missing or
        # already-set). Done at the top so every other resolution below
        # sees the loaded values.
        _load_dotenv_if_present()

        # Resolve API key + base URL — `OFFERGUIDE_LLM_*` is the canonical
        # naming (W12-fix-c). Older `TOKEN/BASE_URL/DEEPSEEK_*/OPENAI_*`
        # still work as fallbacks so existing .env files don't break.
        api_key = (
            os.environ.get("OFFERGUIDE_LLM_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("TOKEN")
            or os.environ.get("OPENAI_API_KEY")
            or None
        )
        base_url = (
            os.environ.get("OFFERGUIDE_LLM_BASE_URL")
            or os.environ.get("DEEPSEEK_BASE_URL")
            or os.environ.get("BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.deepseek.com"
        )
        # Strip stray quote chars (smart-quote copy-paste hazard)
        for ch in ("”", "“", '"', "'", "´"):
            if api_key:
                api_key = api_key.strip().strip(ch)
            base_url = base_url.strip().strip(ch)

        notify_raw = os.environ.get("OFFERGUIDE_NOTIFY", "console").lower()
        notify_channel: NotifyChannel = (
            notify_raw if notify_raw in ("console", "feishu", "telegram") else "console"  # type: ignore[assignment]
        )
        return cls(
            deepseek_api_key=api_key,
            deepseek_base_url=base_url,
            default_model=(
                os.environ.get("OFFERGUIDE_LLM_MODEL")
                or os.environ.get("OFFERGUIDE_DEFAULT_MODEL")
                or os.environ.get("MODEL")
                or "claude-sonnet-4-6"
            ),
            feishu_webhook_url=os.environ.get("FEISHU_WEBHOOK_URL") or None,
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID") or None,
            notify_channel=notify_channel,
            db_path=Path(os.environ.get("OFFERGUIDE_DB", ".offerguide/store.db")),
            resume_pdf=(
                Path(os.environ["OFFERGUIDE_RESUME_PDF"])
                if "OFFERGUIDE_RESUME_PDF" in os.environ
                else None
            ),
            web_host=os.environ.get("OFFERGUIDE_HOST", "127.0.0.1"),
            web_port=int(os.environ.get("OFFERGUIDE_PORT", "8000")),
        )

    def notify_ready(self, channel: NotifyChannel | None = None) -> bool:
        """Whether the picked channel actually has the credentials it needs."""
        ch = channel or self.notify_channel
        if ch == "console":
            return True
        if ch == "feishu":
            return bool(self.feishu_webhook_url)
        if ch == "telegram":
            return bool(self.telegram_bot_token and self.telegram_chat_id)
        return False
