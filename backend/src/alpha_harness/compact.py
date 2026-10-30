"""``python -m alpha_harness.compact``: shrink the local database, once, with Kepler closed.

Trials written before distributions were stored once each (:mod:`.labs.distributions`) carry
every large distribution inline, the same JSON on every trial of a task. This moves each into
``trial_distribution`` once, leaves a reference on the trial, then gives the space back to the
disk. Nothing is lost: every rewritten row is checked to read back exactly as it was before
it is saved, and a row is rewritten in the same transaction that stores its bodies.

It refuses to run while Kepler is open, by the same lock a second backend would trip on.
"""

import asyncio
import json
import sqlite3
import sys
import time
from typing import TYPE_CHECKING, Any

import duckdb

from .config import Settings
from .db.sqlite import Database
from .labs.distributions import pack, unpack

if TYPE_CHECKING:
    from pathlib import Path

#: Trials rewritten per transaction.
BATCH = 500


def _gb(path: Path) -> float:
    total = sum(p.stat().st_size for p in (path, path.with_name(path.name + "-wal")) if p.exists())
    return total / 1e9


def _refuse_if_open(settings: Settings) -> None:
    """Exit if a backend holds the catalog, which it does for as long as it runs."""
    if not settings.duckdb_path.exists():
        return
    try:
        duckdb.connect(str(settings.duckdb_path)).close()
    except duckdb.IOException:
        sys.exit("Kepler is running. Quit it (Ctrl-C in its terminal), then run this again.")


def compact(path: Path, *, out: Any = print) -> dict[str, int]:
    """Move every large inline distribution into ``trial_distribution``. Returns counts."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    total = conn.execute("SELECT count(*) FROM trial").fetchone()[0]
    done = rewritten = 0
    last = 0
    started = time.monotonic()
    while True:
        rows = conn.execute(
            "SELECT id, distributions FROM trial WHERE id > ? ORDER BY id LIMIT ?",
            (last, BATCH),
        ).fetchall()
        if not rows:
            break
        updates: list[tuple[str, int]] = []
        bodies: dict[str, str] = {}
        for trial_id, text in rows:
            raw = json.loads(text) if text else {}
            if not isinstance(raw, dict) or not raw:
                continue
            stored, large = pack(raw)
            if not large:
                continue
            if unpack(stored, {**bodies, **large}) != raw:
                raise RuntimeError(f"Trial {trial_id} would not read back as it was; stopped.")
            bodies.update(large)
            updates.append((json.dumps(stored), trial_id))
        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO trial_distribution (digest, body) VALUES (?, ?)",
                list(bodies.items()),
            )
            conn.executemany("UPDATE trial SET distributions = ? WHERE id = ?", updates)
        rewritten += len(updates)
        done += len(rows)
        last = rows[-1][0]
        rate = done / max(time.monotonic() - started, 1e-6)
        out(f"  {done:,} of {total:,} trials ({done / total:.0%}), {rate:,.0f} a second")
    distinct = conn.execute("SELECT count(*) FROM trial_distribution").fetchone()[0]
    conn.close()
    return {"trials": total, "rewritten": rewritten, "distributions": distinct}


def reclaim(path: Path) -> None:
    """Give the freed pages back to the disk."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()


def main() -> None:
    settings = Settings()
    path = settings.sqlite_path
    if not path.exists():
        sys.exit(f"No database at {path}.")
    _refuse_if_open(settings)
    before = _gb(path)
    print(f"Database: {path} ({before:.1f} GB)")
    print("Bringing the schema up to date…")
    asyncio.run(_migrate(path))
    print("Storing each distribution once…")
    counts = compact(path)
    print(
        f"Rewrote {counts['rewritten']:,} of {counts['trials']:,} trials; "
        f"{counts['distributions']:,} distinct distributions kept."
    )
    print("Giving the space back to the disk (this is the slow part)…")
    reclaim(path)
    print(f"Done: {before:.1f} GB -> {_gb(path):.2f} GB. Start Kepler as usual.")


async def _migrate(path: Path) -> None:
    db = Database(path)
    try:
        await db.create_all()
    finally:
        await db.dispose()


if __name__ == "__main__":
    main()
