"""W14 → W15 migration: agent_self_notes (SQL) → worldview/MEMORY.md (markdown).

Why a script and not auto on first W15 run? Because the agent owns
worldview after W15 takes over. Auto-migration on every startup would
risk re-writing notes the agent has already cleared/restructured.

Usage:
    uv run python scripts/migrate_w14_to_w15.py           # dry-run preview
    uv run python scripts/migrate_w14_to_w15.py --apply   # actually write

Idempotence: marks migrated rows with cleared_reason='migrated to W15
worldview' so re-running is a no-op.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

# Allow running from repo root via `python scripts/migrate_w14_to_w15.py`
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from offerguide.config import Settings  # noqa: E402
from offerguide.harness import default_worldview_dir  # noqa: E402
from offerguide.harness.memory import MemoryStore  # noqa: E402
from offerguide.memory import Store  # noqa: E402

_MIGRATED_REASON = "migrated to W15 worldview"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually write (default: dry-run preview)")
    args = ap.parse_args()

    settings = Settings.from_env()
    store = Store(settings.db_path)

    # If table doesn't exist, this is a fresh DB → nothing to migrate.
    with store.connect() as conn:
        try:
            rows = conn.execute(
                "SELECT id, body, note_kind, created_at "
                "FROM agent_self_notes "
                "WHERE cleared_at IS NULL "
                "ORDER BY created_at ASC"
            ).fetchall()
        except Exception as e:
            print(f"agent_self_notes table not found ({e}). "
                  "Nothing to migrate (fresh DB).")
            return 0

    if not rows:
        print("No active agent_self_notes rows. Nothing to migrate.")
        return 0

    print(f"Found {len(rows)} active self_notes rows.")
    for row in rows:
        nid, body, kind, created = int(row[0]), row[1], row[2], row[3]
        print(f"  - #{nid} [{kind}] (created {created}): {body[:80]}")

    if not args.apply:
        print("\n(dry-run; pass --apply to actually migrate)")
        return 0

    # Build markdown block to append to MEMORY.md
    today = _dt.date.today().isoformat()
    lines = [
        "",
        f"## 从 W14 self_notes 迁移过来 ({today})",
        "",
    ]
    for row in rows:
        body, kind = row[1], row[2]
        prefix = {"todo": "[ ]", "observation": "👁", "context": "📝"}.get(
            kind or "", "·"
        )
        lines.append(f"{prefix} {body}")
    block = "\n".join(lines) + "\n"

    # Append to MEMORY.md (or create if doesn't exist)
    wdir = default_worldview_dir(settings)
    mem = MemoryStore(root=wdir)
    memory_md = wdir / "MEMORY.md"
    existing = memory_md.read_text(encoding="utf-8") if memory_md.exists() else ""
    new_text = existing.rstrip() + "\n" + block

    result = mem.execute({
        "command": "create",
        "path": "MEMORY.md",
        "file_text": new_text,
    })
    print(f"\nMEMORY.md write: {result}")

    # Mark migrated rows
    with store.connect() as conn:
        for row in rows:
            conn.execute(
                "UPDATE agent_self_notes SET cleared_at = julianday('now'), "
                "cleared_reason = ? WHERE id = ?",
                (_MIGRATED_REASON, int(row[0])),
            )
    print(f"Marked {len(rows)} rows as cleared (reason={_MIGRATED_REASON!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
