#!/usr/bin/env python3
"""Merge two staging queue SQLite databases into one.

Usage:
    # Step 1: download your existing Kaggle queue.db to data/queue.db
    #         (Kaggle → Datasets → aesthetic-pairs-queue → Download)
    # Step 2:
    py -3.10 scripts/merge_queue.py

Output: data/merged_queue.db — upload this to Kaggle as the new queue.db
"""
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime

OLD_DB      = Path("data/queue.db")            # downloaded from Kaggle
NEW_DB      = Path("data/staging.db")          # produced by local crawler
REALSELF_DB = Path("data/realself_pairs.db")   # downloaded from Colab
OUT_DB      = Path("data/merged_queue.db")     # upload this to Kaggle

def main():
    if not OLD_DB.exists():
        print(f"ERROR: {OLD_DB} not found.")
        print("Download your Kaggle queue.db to data/queue.db first:")
        print("  Go to kaggle.com → Datasets → aesthetic-pairs-queue → Download")
        return

    # Start from a copy of the old DB so we keep all original pairs
    shutil.copy2(OLD_DB, OUT_DB)
    print(f"Copied {OLD_DB} → {OUT_DB}")

    conn = sqlite3.connect(OUT_DB)
    conn.execute("PRAGMA journal_mode=WAL")

    # Count before
    before = conn.execute("SELECT COUNT(*) FROM staging_queue").fetchone()[0]
    print(f"Old queue: {before} pairs")

    def merge_db(path: Path, label: str) -> int:
        if not path.exists():
            print(f"Skipping {label} ({path} not found)")
            return 0
        conn.execute(f"ATTACH DATABASE '{path}' AS extra_{label}")
        conn.execute(f"""
            INSERT OR IGNORE INTO main.staging_queue
                (before_url, after_url, source_url, source_name,
                 language, consent_tier, metadata, status, created_at)
            SELECT
                before_url, after_url, source_url,
                COALESCE(source_name, '{label}'),
                COALESCE(language, 'en'),
                COALESCE(consent_tier, 2),
                metadata, 'pending',
                COALESCE(created_at, ?)
            FROM extra_{label}.staging_queue
            WHERE status = 'pending'
        """, (datetime.utcnow().isoformat(),))
        conn.execute(f"DETACH DATABASE extra_{label}")
        conn.commit()
        after_merge = conn.execute("SELECT COUNT(*) FROM staging_queue").fetchone()[0]
        added = after_merge - before
        print(f"  + {label}: {added} new pairs")
        return added

    # Merge local crawler output
    merge_db(NEW_DB, "local")
    # Merge RealSelf Colab output (if present)
    merge_db(REALSELF_DB, "realself")

    # Count after
    after = conn.execute("SELECT COUNT(*) FROM staging_queue").fetchone()[0]
    added = after - before

    # Label coverage
    labeled = conn.execute("""
        SELECT COUNT(*) FROM staging_queue
        WHERE metadata LIKE '%treatment_category%' AND status='pending'
    """).fetchone()[0]

    conn.close()

    print(f"New pairs added: {added}")
    print(f"Total pairs:     {after}")
    print(f"Labeled pairs:   {labeled} ({100*labeled//max(after,1)}%)")
    print(f"\nUpload {OUT_DB} to Kaggle as the new queue.db:")
    print("  kaggle.com → Datasets → aesthetic-pairs-queue → New Version → upload merged_queue.db")
    print("  (rename to queue.db when uploading)")

if __name__ == "__main__":
    main()
