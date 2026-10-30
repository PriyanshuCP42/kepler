"""Distributions stored once each: the stored form reads back exactly, old rows compact."""

import json
import sqlite3

from alpha_harness.compact import compact
from alpha_harness.labs.distributions import INLINE_LIMIT, REF, pack, refs, unpack

FIELDS = json.dumps(
    {"name": "CategoricalDistribution", "attributes": {"choices": [f"f{i}" for i in range(400)]}}
)
LOOKBACK = json.dumps({"name": "CategoricalDistribution", "attributes": {"choices": [5, 21]}})


def test_large_distributions_become_references_and_read_back():
    raw = {"field": FIELDS, "lookback": LOOKBACK, "field_B": "space"}
    stored, bodies = pack(raw)
    assert stored["field"].startswith(REF)
    assert stored["lookback"] == LOOKBACK  # short ones stay inline
    assert stored["field_B"] == "space"
    assert len(json.dumps(stored)) < INLINE_LIMIT
    assert unpack(stored, bodies) == raw
    assert refs([stored]) == set(bodies)


def test_a_missing_body_makes_the_trial_unreplayable_not_wrong():
    stored, _ = pack({"field": FIELDS})
    assert unpack(stored, {}) == {}


def test_packing_twice_changes_nothing():
    stored, _ = pack({"field": FIELDS})
    again, bodies = pack(stored)
    assert again == stored
    assert bodies == {}


def test_compact_rewrites_old_rows_and_keeps_one_copy(tmp_path):
    path = tmp_path / "h.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trial (id INTEGER PRIMARY KEY, distributions JSON)")
    conn.execute(
        "CREATE TABLE trial_distribution (digest VARCHAR(64) PRIMARY KEY, body TEXT NOT NULL)"
    )
    originals = {
        i: {"field": FIELDS, "lookback": LOOKBACK} if i % 3 else {} for i in range(1, 1201)
    }
    conn.executemany(
        "INSERT INTO trial (id, distributions) VALUES (?, ?)",
        [(i, json.dumps(d)) for i, d in originals.items()],
    )
    conn.commit()
    conn.close()

    counts = compact(path, out=lambda _: None)
    assert counts["rewritten"] == sum(1 for d in originals.values() if d)
    assert counts["distributions"] == 1

    conn = sqlite3.connect(path)
    bodies = dict(conn.execute("SELECT digest, body FROM trial_distribution").fetchall())
    for trial_id, text in conn.execute("SELECT id, distributions FROM trial"):
        assert unpack(json.loads(text), bodies) == originals[trial_id]
    conn.close()
    # A second run finds nothing left to do.
    assert compact(path, out=lambda _: None)["rewritten"] == 0
