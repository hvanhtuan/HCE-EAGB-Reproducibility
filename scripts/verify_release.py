"""Fail-fast verification for the public reproducibility release."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ERRORS: list[str] = []


def check(condition: bool, message: str) -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {message}")
    if not condition:
        ERRORS.append(message)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    required = [
        "README.md", "LICENSE", "CITATION.cff", "Dockerfile",
        "requirements-lock.txt", "configs/seeds.json", "configs/paper.json",
        "data/raw_manifest.json", "results/prereg_H1.json",
        "results/reference/kb2_test.json",
        "results/supplementary/kb2_pilot.json",
        "results/audit/matching_audit_summary.json",
        "artifacts/model_selection/model_selection.csv",
        "scripts/run_all.py", "scripts/export_cell_predictions.py",
        "figures/build_all.py", "SHA256SUMS.txt",
    ]
    for relative in required:
        check((ROOT / relative).is_file(), f"required file: {relative}")

    json_files = list((ROOT / "configs").glob("*.json"))
    json_files += list((ROOT / "results" / "reference").glob("*.json"))
    json_files += list((ROOT / "results" / "supplementary").glob("*.json"))
    json_files += list((ROOT / "results" / "audit").glob("*.json"))
    parsed = 0
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
            parsed += 1
        except Exception as exc:
            ERRORS.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
    check(parsed == len(json_files), f"JSON parse: {parsed}/{len(json_files)}")

    seed = json.loads((ROOT / "configs" / "seeds.json").read_text(encoding="utf-8"))
    check(seed.get("global") == 20260918, "central seed matches historical core seed")

    prereg = json.loads((ROOT / "results" / "prereg_H1.json").read_text(encoding="utf-8"))
    check(prereg.get("data_split", {}).get("test") == [2021, 2023], "locked test window is 2021-2023")
    check(bool(prereg.get("k_selected")) and bool(prereg.get("rounds")), "locked model configuration present")

    selection_path = ROOT / "artifacts" / "model_selection" / "model_selection.csv"
    if selection_path.exists():
        with selection_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        check(len(rows) >= 30, f"model-selection log has {len(rows)} rows")
        check(any(row.get("selected") == "true" for row in rows), "model-selection log records selected candidates")

    prediction_path = ROOT / "artifacts" / "predictions" / "nfip_test_predictions.parquet"
    schema_path = prediction_path.with_suffix(".schema.json")
    if prediction_path.exists():
        try:
            import pandas as pd
            frame = pd.read_parquet(prediction_path)
            check(len(frame) == 399128, f"cell-level prediction row count: {len(frame)}")
            check(frame["cell_id"].is_unique, "cell-level prediction IDs are unique")
            check(any(column.startswith("prediction_") for column in frame), "prediction columns present")
            check(schema_path.exists(), "prediction schema present")
            if schema_path.exists():
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
                check(schema.get("sha256") == sha256(prediction_path), "prediction SHA-256 matches schema")
        except Exception as exc:
            check(False, f"prediction artifact readable: {exc}")
    else:
        print("[INFO] Cell-level Parquet is not committed; run the full replay to generate it.")

    checksum_path = ROOT / "SHA256SUMS.txt"
    checked = 0
    if checksum_path.exists():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = ROOT / Path(relative)
            if not path.exists():
                ERRORS.append(f"checksum target missing: {relative}")
                continue
            if sha256(path) != expected:
                ERRORS.append(f"checksum mismatch: {relative}")
                continue
            checked += 1
        check(checked > 0, f"verified {checked} release checksums")

    oversized = [path for path in ROOT.rglob("*") if path.is_file() and path.stat().st_size > 95 * 1024 * 1024]
    check(not oversized, "no tracked-style file exceeds GitHub's 100 MB limit")

    print("\n" + ("Release verification PASSED." if not ERRORS else "Release verification FAILED."))
    if ERRORS:
        for error in ERRORS:
            print(f" - {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
