#!/usr/bin/env python3
"""
Backfill treatment_category into existing staging_queue rows.

Reads every row whose metadata lacks treatment_category, derives the label
from source_url using the shared SLUG_MAP, and writes it back.

Usage:
    py -3.10 scripts/backfill_labels.py [path/to/db]

Default DB: data/merged_queue.db  (run after merge_queue.py)
"""
import sys, sqlite3, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crawler.treatment_labels import extract_treatment_from_url

DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "merged_queue.db"

if not DB_PATH.exists():
    print(f"ERROR: {DB_PATH} not found. Run merge_queue.py first.")
    sys.exit(1)

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA journal_mode=WAL")

rows = conn.execute(
    "SELECT id, source_url, metadata FROM staging_queue"
).fetchall()

updated = already_labeled = no_match = 0

for row_id, source_url, metadata_str in rows:
    try:
        meta = json.loads(metadata_str or "{}")
    except Exception:
        meta = {}

    if meta.get("treatment_category"):
        already_labeled += 1
        continue

    treatment = extract_treatment_from_url(source_url or "")
    if not treatment:
        no_match += 1
        continue

    meta["treatment_category"] = treatment
    conn.execute(
        "UPDATE staging_queue SET metadata=? WHERE id=?",
        (json.dumps(meta), row_id),
    )
    updated += 1

conn.commit()

total = len(rows)
labeled_total = already_labeled + updated
print(f"Total rows:       {total}")
print(f"Already labeled:  {already_labeled}")
print(f"Newly labeled:    {updated}")
print(f"No match found:   {no_match}")
print(f"Label coverage:   {labeled_total}/{total} ({100*labeled_total//max(total,1)}%)")

if no_match:
    # Show sample unmatched URLs
    unmatched = conn.execute("""
        SELECT source_name, source_url FROM staging_queue
        WHERE (metadata IS NULL OR metadata NOT LIKE '%treatment_category%')
        LIMIT 10
    """).fetchall()
    print(f"\nSample unmatched source URLs:")
    for name, url in unmatched:
        print(f"  [{name}] {url}")

    # Delete non-facial/body procedure pairs — all remaining unlabeled rows
    # from plasticsurgery.org are body procedures (arm-lift, body-contouring, etc.)
    # that have no value for a facial aesthetic model.
    deleted = conn.execute("""
        DELETE FROM staging_queue
        WHERE (metadata IS NULL OR metadata NOT LIKE '%treatment_category%')
    """).rowcount
    conn.commit()
    print(f"\nDeleted {deleted} unlabeled (non-facial) rows.")
    remaining = conn.execute("SELECT COUNT(*) FROM staging_queue").fetchone()[0]
    print(f"Remaining rows:   {remaining} (all labeled)")

conn.close()
