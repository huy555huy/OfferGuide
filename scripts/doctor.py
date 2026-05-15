#!/usr/bin/env python3
"""OfferGuide setup doctor — one-shot environment diagnosis.

Fixes the W15.13-review "装机 30 分钟" complaint. Run before the server
starts; for each check, print PASS / FAIL with **specific remediation**.

Usage:
    uv run python scripts/doctor.py [--quick] [--probe-llm]

Exit code 0 = green; non-zero = at least one critical issue.

Checks (in order — earliest failures hide later ones):
1. Python version (≥ 3.11)
2. uv installed
3. Project deps (offerguide importable)
4. .env file present + readable
5. DEEPSEEK_API_KEY set + minimum sanity (length / format)
6. TAVILY_API_KEY set (optional — only warn, not fail)
7. OFFERGUIDE_DB writable directory
8. Worldview dir writable (creates if missing)
9. Resume file readable (if OFFERGUIDE_RESUME_PDF set)
10. Port 8000 free (or whatever OFFERGUIDE_PORT)
11. SQLite schema migrations applied
12. (--probe-llm) Live ping to LLM API + Tavily

Each check prints:
- ``✓ <name>`` (green)
- ``✗ <name>: <what's wrong>. Fix: <how>``
- ``⚠ <name>: <warning>``
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

# Colours (no external dep — plain ANSI)
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _green(msg: str) -> None:
    print(f"{_GREEN}✓{_RESET} {msg}")


def _red(msg: str, fix: str = "") -> None:
    print(f"{_RED}✗{_RESET} {msg}")
    if fix:
        print(f"  {_BOLD}Fix:{_RESET} {fix}")


def _warn(msg: str, fix: str = "") -> None:
    print(f"{_YELLOW}⚠{_RESET} {msg}")
    if fix:
        print(f"  {_DIM}Suggest:{_RESET} {fix}")


def _info(msg: str) -> None:
    print(f"  {_DIM}{msg}{_RESET}")


def check_python() -> bool:
    v = sys.version_info
    if v < (3, 11):
        _red(
            f"Python version: {v.major}.{v.minor}.{v.micro} (need ≥ 3.11)",
            "Install Python 3.11+ via pyenv: `pyenv install 3.11.9 && pyenv local 3.11.9`",
        )
        return False
    _green(f"Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_uv() -> bool:
    import shutil
    if shutil.which("uv") is None:
        _red(
            "uv not on PATH",
            "Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`",
        )
        return False
    _green("uv installed")
    return True


def check_offerguide_importable() -> bool:
    try:
        import offerguide  # noqa: F401
        _green("offerguide package importable")
        return True
    except ImportError as e:
        _red(
            f"offerguide not importable: {e}",
            "Run: `uv sync --extra ui --extra autonomous`",
        )
        return False


def check_dotenv(repo_root: Path) -> bool:
    env_file = repo_root / ".env"
    if not env_file.exists():
        _red(
            ".env file missing at repo root",
            "Create .env with at least DEEPSEEK_API_KEY. "
            "Template: `cp .env.example .env` (if exists), or write 1 line: "
            '`DEEPSEEK_API_KEY="sk-..."`',
        )
        return False
    _green(f".env found ({env_file.stat().st_size}B)")
    return True


def _load_env(repo_root: Path) -> None:
    """Best-effort .env loader (don't depend on python-dotenv if not installed)."""
    env_file = repo_root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def check_deepseek_key() -> bool:
    # OFFERGUIDE_LLM_API_KEY is canonical; fall back to legacy names per
    # config.py order. Same resolution Settings.from_env() uses.
    key_var = None
    for name in (
        "OFFERGUIDE_LLM_API_KEY", "DEEPSEEK_API_KEY", "TOKEN", "OPENAI_API_KEY",
    ):
        if os.environ.get(name):
            key_var = name
            break
    key = os.environ.get(key_var or "", "")
    if not key:
        _red(
            "LLM API key not set (looked for OFFERGUIDE_LLM_API_KEY, DEEPSEEK_API_KEY, TOKEN, OPENAI_API_KEY)",
            "Get a key at https://platform.deepseek.com/api_keys (free tier OK). "
            'Add to .env: `OFFERGUIDE_LLM_API_KEY="sk-..."`',
        )
        return False
    if len(key) < 20:
        _red(
            f"{key_var} looks malformed (len={len(key)})",
            "Re-copy from https://platform.deepseek.com/api_keys",
        )
        return False
    _green(f"LLM API key set via {key_var} ({len(key)} chars, starts {key[:6]}...)")
    return True


def check_tavily_key() -> bool:
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        _warn(
            "TAVILY_API_KEY not set (discover_jobs / web_search will be disabled)",
            "Get a key at https://tavily.com/ (free tier 1000 calls/mo)",
        )
        return True  # not fatal — agent can still help with pasted JDs
    _green(f"TAVILY_API_KEY set ({len(key)} chars)")
    return True


def check_db_path(repo_root: Path) -> bool:
    from offerguide.config import Settings
    settings = Settings.from_env()
    db_path = Path(settings.db_path)
    if not db_path.is_absolute():
        db_path = repo_root / db_path
    parent = db_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _red(
            f"DB parent dir not creatable: {parent} ({e})",
            f"Run: `mkdir -p {parent}` and check permissions",
        )
        return False
    if db_path.exists() and not os.access(db_path, os.W_OK):
        _red(
            f"DB file not writable: {db_path}",
            f"Fix permissions: `chmod 644 {db_path}`",
        )
        return False
    _green(f"DB path OK: {db_path}")
    return True


def check_worldview_dir() -> bool:
    from offerguide.agent_runtime import default_worldview_dir
    wdir = default_worldview_dir()
    try:
        wdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _red(
            f"Worldview dir not creatable: {wdir} ({e})",
            "Override with `OFFERGUIDE_WORLDVIEW_DIR=...` in .env",
        )
        return False
    test_f = wdir / ".doctor-write-test"
    try:
        test_f.write_text("ok")
        test_f.unlink()
    except OSError as e:
        _red(
            f"Worldview dir not writable: {wdir} ({e})",
            "Check directory permissions",
        )
        return False
    _green(f"Worldview dir OK: {wdir}")
    return True


def check_resume_file() -> bool:
    from offerguide.config import Settings
    settings = Settings.from_env()
    resume = settings.resume_pdf
    if resume is None:
        _warn(
            "OFFERGUIDE_RESUME_PDF not set (score_match / tailor_advice will fail)",
            'Add to .env: `OFFERGUIDE_RESUME_PDF="/path/to/中文简历.pdf"` '
            "(or .docx — both supported)",
        )
        return True
    p = Path(resume)
    if not p.exists():
        _red(
            f"Resume file not found: {p}",
            "Check the path in .env, or update OFFERGUIDE_RESUME_PDF",
        )
        return False
    if not os.access(p, os.R_OK):
        _red(
            f"Resume file not readable: {p}",
            f"Fix permissions: `chmod 644 {p}`",
        )
        return False
    sz = p.stat().st_size
    _green(f"Resume file OK: {p.name} ({sz}B)")
    return True


def check_port_free() -> bool:
    from offerguide.config import Settings
    settings = Settings.from_env()
    port = settings.web_port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        sock.close()
        _green(f"Port {port} free")
        return True
    except OSError:
        _warn(
            f"Port {port} in use (server already running?)",
            "Stop the existing server, or set OFFERGUIDE_PORT to a free port",
        )
        return True  # not fatal — might be intentional


def check_db_schema() -> bool:
    try:
        from offerguide import Store
        from offerguide.config import Settings
        from offerguide.agent_runtime import _schema as harness_schema
        settings = Settings.from_env()
        store = Store(settings.db_path)
        store.init_schema()
        harness_schema.init_agent_runtime_schema(store)
        with store.connect() as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        required = {"jobs", "applications", "skill_runs", "harness_runs",
                    "harness_events", "harness_scheduled_wakes",
                    "evolution_signals", "inbox_items"}
        missing = required - tables
        if missing:
            _red(
                f"Missing tables: {sorted(missing)}",
                "Run `uv run python scripts/doctor.py` again — schema init "
                "should have created them. If still missing, file a bug.",
            )
            return False
        _green(f"Schema OK ({len(tables)} tables, including {len(required)} required)")
        return True
    except Exception as e:
        _red(
            f"Schema check crashed: {e}",
            "DB file might be corrupt. Backup + delete `.offerguide/store.db` "
            "to start fresh (you'll lose dogfood data).",
        )
        return False


def probe_llm() -> bool:
    """Live ping — only when --probe-llm passed (costs ~$0.001)."""
    from offerguide.config import Settings
    from offerguide.llm import LLMClient, LLMError
    settings = Settings.from_env()
    if not settings.deepseek_api_key:
        _warn("Skipping LLM probe (no API key)", "")
        return True
    try:
        client = LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )
        resp = client.chat(
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.0,
        )
        cost = resp.cost_usd or 0.0
        _green(
            f"LLM live ping OK ({resp.model}, {resp.prompt_tokens}+{resp.completion_tokens} tokens, ${cost:.5f})"
        )
        return True
    except LLMError as e:
        _red(
            f"LLM probe failed: {e}",
            "Check DEEPSEEK_API_KEY validity at https://platform.deepseek.com/usage",
        )
        return False


def probe_tavily() -> bool:
    if not os.environ.get("TAVILY_API_KEY"):
        return True  # skipped earlier with warn
    try:
        from offerguide.agentic.search import build_default_search
        backend = build_default_search()
        hits = backend.search("test query", max_results=1)
        _green(f"Tavily live ping OK ({len(hits)} hit returned)")
        return True
    except Exception as e:
        _warn(
            f"Tavily probe failed: {e}",
            "Check TAVILY_API_KEY at https://tavily.com/dashboard",
        )
        return True  # non-fatal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip schema + import checks (fastest)",
    )
    parser.add_argument(
        "--probe-llm", action="store_true",
        help="Make a live LLM API call (~$0.001) to verify the key works",
    )
    args = parser.parse_args()

    print(f"{_BOLD}OfferGuide doctor{_RESET} — checking environment...")
    print()

    # Find repo root (we're in scripts/, so parent is repo root)
    repo_root = Path(__file__).parent.parent.resolve()
    os.chdir(repo_root)
    _info(f"repo root: {repo_root}")
    print()

    # Load .env BEFORE any check that depends on env vars
    _load_env(repo_root)

    failures = 0
    checks = [
        ("python", check_python),
        ("uv", check_uv),
    ]
    if not args.quick:
        checks.append(("offerguide importable", check_offerguide_importable))
    checks.extend([
        (".env file", lambda: check_dotenv(repo_root)),
        ("DEEPSEEK_API_KEY", check_deepseek_key),
        ("TAVILY_API_KEY", check_tavily_key),
        ("DB path", lambda: check_db_path(repo_root)),
        ("worldview dir", check_worldview_dir),
        ("resume file", check_resume_file),
        ("web port", check_port_free),
    ])
    if not args.quick:
        checks.append(("DB schema", check_db_schema))
    if args.probe_llm:
        checks.append(("LLM live ping", probe_llm))
        checks.append(("Tavily live ping", probe_tavily))

    for _name, fn in checks:
        try:
            ok = fn()
            if not ok:
                failures += 1
        except Exception as e:
            _red(f"{_name} crashed: {type(e).__name__}: {e}",
                 "File a bug — this check should never raise")
            failures += 1

    print()
    if failures == 0:
        print(f"{_GREEN}{_BOLD}All green!{_RESET} You can start the server:")
        print(f"  {_DIM}uv run --extra ui python -m offerguide.ui.web{_RESET}")
        return 0
    print(
        f"{_RED}{_BOLD}{failures} issue(s){_RESET} — fix above, then re-run "
        f"`uv run python scripts/doctor.py`"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
