"""W19+ audit — verify FastAPI lifespan really starts the ambient daemon.

Goal (audit doc item #1): prove `_ambient_discovery_loop` becomes a real
asyncio.Task when TestClient enters the lifespan context, and that the task
gets cancelled cleanly on exit.

We don't want to actually run a 30s + crawl cycle in the test, so:
  - Monkeypatch `_ambient_discovery_loop` with a no-op coroutine that records
    it was called + then sleeps until cancelled.
  - Enter TestClient context (triggers lifespan startup).
  - Assert: our patched coroutine was called.
  - Exit TestClient context (triggers lifespan shutdown).
  - Assert: task is done (cancellation propagated).

Run:
    PYTHONPATH=src python scripts/audit_w19_lifespan_boots_daemon.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import offerguide
from offerguide.config import Settings
from offerguide.agent_runtime import _schema as harness_schema
from offerguide.profile.schema import UserProfile

# Capture state across the patched coroutine
state: dict = {"called": False, "kwargs": None, "cancel_seen": False, "task_alive": None}

async def _fake_loop(*args, **kwargs):
    """Stand-in for ambient daemon loop. Records call, then waits for cancel."""
    state["called"] = True
    state["kwargs"] = list(kwargs.keys())
    try:
        # Sleep way longer than the TestClient context so we can detect cancel
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        state["cancel_seen"] = True
        raise

# Patch BEFORE importing/calling create_app — create_app does
# `from ..workers.ambient import _ambient_discovery_loop` at lifespan-build
# time, so the patch must be at the module attribute it'll resolve to.
import offerguide.workers.ambient as _amb_mod
_amb_mod._ambient_discovery_loop = _fake_loop  # type: ignore[assignment]

from offerguide.ui.web import create_app  # noqa: E402  (after patch)

tmp = Path(tempfile.mkdtemp(prefix="ogfd_w19_lifespan_"))
db = tmp / "store.db"
store = offerguide.Store(db)
store.init_schema()
harness_schema.init_agent_runtime_schema(store)

profile = UserProfile(raw_resume_text="测试简历")

# Critical: a non-empty key + disable_ambient_crawl=False → lifespan must
# create the bg task. Use a fake key string (won't be used since we patched
# the loop coroutine).
settings = Settings(
    deepseek_api_key="sk-fake-for-lifespan-test",
    db_path=db,
    disable_ambient_crawl=False,
)

app = create_app(
    settings=settings, store=store, profile=profile,
    skills=[], runtime=None, notifier=None,
)

print("Built app. Entering TestClient context (will fire lifespan startup)...")
with TestClient(app) as client:
    # Hit a trivial endpoint to ensure the app is fully booted
    r = client.get("/")
    print(f"  GET / → {r.status_code}")
    # At this point lifespan startup has run. Check our flag.
    print(f"  state.called: {state['called']}")
    print(f"  state.kwargs: {state['kwargs']}")
    # Find the bg task
    loop = asyncio.new_event_loop()
    # We can't easily inspect the running loop's tasks here from sync code,
    # but the `state.called` flag is direct evidence the coroutine ran.

print("Exited TestClient context (lifespan shutdown ran)...")
print(f"  state.cancel_seen: {state['cancel_seen']}")

# ---- assertions --------------------------------------------------------

print("\n" + "=" * 60)
ok = True

if not state["called"]:
    print("✗ FAIL: ambient loop coroutine was NEVER called by lifespan")
    ok = False
else:
    print("✓ PASS: ambient loop coroutine was called at startup")

expected_kwargs = {"store", "settings", "runtime", "skills", "user_profile_text"}
got_kwargs = set(state["kwargs"] or [])
if expected_kwargs - got_kwargs:
    print(f"✗ FAIL: lifespan call missing kwargs: {expected_kwargs - got_kwargs}")
    ok = False
else:
    print(f"✓ PASS: lifespan passed all expected kwargs ({got_kwargs})")

if not state["cancel_seen"]:
    print("✗ FAIL: bg task was NOT cancelled on shutdown (will leak)")
    ok = False
else:
    print("✓ PASS: bg task received CancelledError on shutdown (clean teardown)")

# Now verify the OPPOSITE: when key is empty OR disable_ambient_crawl=True,
# loop must NOT be started.
print("\n" + "-" * 60)
print("Negative case 1: empty key → loop should NOT start")

state2: dict = {"called": False}
async def _fake_loop2(*args, **kwargs):
    state2["called"] = True
    await asyncio.sleep(3600)
_amb_mod._ambient_discovery_loop = _fake_loop2  # type: ignore[assignment]

settings_no_key = Settings(deepseek_api_key=None, db_path=db, disable_ambient_crawl=False)
app2 = create_app(
    settings=settings_no_key, store=store, profile=profile,
    skills=[], runtime=None, notifier=None,
)
with TestClient(app2) as c:
    c.get("/")
print(f"  state2.called: {state2['called']}")
if state2["called"]:
    print("✗ FAIL: loop started even with no key")
    ok = False
else:
    print("✓ PASS: loop correctly skipped when no LLM key")

print("\nNegative case 2: disable_ambient_crawl=True → loop should NOT start")
state3: dict = {"called": False}
async def _fake_loop3(*args, **kwargs):
    state3["called"] = True
    await asyncio.sleep(3600)
_amb_mod._ambient_discovery_loop = _fake_loop3  # type: ignore[assignment]

settings_disabled = Settings(
    deepseek_api_key="sk-fake", db_path=db, disable_ambient_crawl=True,
)
app3 = create_app(
    settings=settings_disabled, store=store, profile=profile,
    skills=[], runtime=None, notifier=None,
)
with TestClient(app3) as c:
    c.get("/")
print(f"  state3.called: {state3['called']}")
if state3["called"]:
    print("✗ FAIL: loop started even with disable_ambient_crawl=True")
    ok = False
else:
    print("✓ PASS: loop correctly skipped when disable_ambient_crawl=True")

print("=" * 60)
sys.exit(0 if ok else 1)
