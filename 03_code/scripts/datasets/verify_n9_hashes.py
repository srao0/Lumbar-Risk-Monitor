#!/usr/bin/env python3
"""Confirm the n=9 frozen feature CSVs match the SHA-256 recorded in evaluation_summary.json.

Run from anywhere:  python scripts/verify_n9_hashes.py
Exit 0 = all match (byte-consistent freeze); exit 1 = mismatch (re-run refreeze_n9.py).
Needs only base Python (no sklearn/pandas).
"""
import hashlib, json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results" / "fallback_analysis_sets_n9"
summary = R / "evaluation_corrected" / "evaluation_summary.json"
s = json.loads(summary.read_text(encoding="utf-8-sig"))

all_ok = True
for name, recorded in s["sha256"].items():
    f = R / name
    disk = hashlib.sha256(f.read_bytes()).hexdigest()
    ok = (disk == recorded); all_ok &= ok
    print(f"{'OK  ' if ok else 'FAIL'}  {name}")
    print(f"         disk     {disk}")
    print(f"         recorded {recorded}")
print()
print("ALL MATCH - n=9 freeze is byte-consistent. Done." if all_ok
      else "MISMATCH - re-run scripts/refreeze_n9.py (write to a LOCAL non-OneDrive folder, then copy the two CSVs in).")
raise SystemExit(0 if all_ok else 1)
