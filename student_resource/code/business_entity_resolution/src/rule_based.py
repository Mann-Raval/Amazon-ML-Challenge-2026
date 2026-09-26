"""Memory-conscious deterministic baseline for the entity-resolution challenge.

The large Source-2/3 files are indexed in SQLite on disk. Only one Source-1
row is held at a time, so this baseline is suitable for machines that cannot
hold the full Python token index in RAM.
"""
import argparse
import csv
import os
import sqlite3
import tempfile

from .normalize import normalize_name, normalize_address


def add_source(conn, path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f, delimiter="\t")
        batch = []
        for row in rows:
            name, _ = normalize_name(row.get("business_name", ""))
            addr, _tokens, postal = normalize_address(row.get("business_address", ""))
            country = (row.get("country") or "").strip().lower()
            batch.append((country, name, addr, postal or "", row["entity_id"]))
            if len(batch) >= 10_000:
                conn.executemany("INSERT INTO records VALUES (?,?,?,?,?)", batch)
                batch.clear()
        if batch:
            conn.executemany("INSERT INTO records VALUES (?,?,?,?,?)", batch)
    conn.commit()


def run(source1, source2, source3, matching_out, candidate_out, db_path=None):
    temporary = db_path is None
    if temporary:
        handle, db_path = tempfile.mkstemp(prefix="entity_resolution_", suffix=".sqlite")
        os.close(handle)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("CREATE TABLE IF NOT EXISTS records (country TEXT, name TEXT, addr TEXT, postal TEXT, eid TEXT)")
        if conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0:
            add_source(conn, source2)
            add_source(conn, source3)
            conn.execute("CREATE INDEX IF NOT EXISTS records_name ON records(country,name)")
            conn.execute("CREATE INDEX IF NOT EXISTS records_addr ON records(country,addr)")
            conn.execute("CREATE INDEX IF NOT EXISTS records_postal ON records(country,postal)")
            conn.commit()

        os.makedirs(os.path.dirname(matching_out) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(candidate_out) or ".", exist_ok=True)
        with open(source1, encoding="utf-8", newline="") as f1, \
             open(matching_out, "w", encoding="utf-8", newline="") as fm, \
             open(candidate_out, "w", encoding="utf-8", newline="") as fc:
            rows = csv.DictReader(f1, delimiter="\t")
            mw, cw = csv.writer(fm, delimiter="\t", lineterminator="\n"), csv.writer(fc, delimiter="\t", lineterminator="\n")
            mw.writerow(["source1_entity_id", "matched_entity_ids"])
            cw.writerow(["source1_entity_id", "candidate_entity_ids"])
            for row in rows:
                eid = row["entity_id"]
                name, _ = normalize_name(row.get("business_name", ""))
                addr, _tokens, postal = normalize_address(row.get("business_address", ""))
                country = (row.get("country") or "").strip().lower()
                # Strong rules: exact normalized name plus exact address or PIN;
                # allow name-only only when it identifies one pool record.
                candidates = set()
                if name:
                    candidates.update(x[0] for x in conn.execute(
                        "SELECT eid FROM records WHERE country=? AND name=?", (country, name)))
                strong = set()
                for cid in candidates:
                    rec = conn.execute("SELECT addr,postal FROM records WHERE eid=?", (cid,)).fetchone()
                    if rec and ((addr and rec[0] == addr) or (postal and rec[1] == postal)):
                        strong.add(cid)
                if not strong and len(candidates) == 1:
                    strong = candidates
                ordered = sorted(strong)
                mw.writerow([eid, ",".join(ordered)])
                cw.writerow([eid, ",".join(sorted(candidates))])
    finally:
        conn.close()
        if temporary:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(db_path + suffix)
                except FileNotFoundError:
                    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", required=True)
    p.add_argument("--source2", required=True)
    p.add_argument("--source3", required=True)
    p.add_argument("--matching-out", required=True)
    p.add_argument("--candidate-out", required=True)
    p.add_argument("--db", default=None, help="Optional persistent SQLite index path")
    a = p.parse_args()
    run(a.source1, a.source2, a.source3, a.matching_out, a.candidate_out, a.db)


if __name__ == "__main__":
    main()
