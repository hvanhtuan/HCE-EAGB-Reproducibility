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
        "results/supplementary/numerical_stability.json",
        "results/supplementary/postreview_ablation_windows.json",
        "results/supplementary/missing_ded_sensitivity.json",
        "results/replay_verification.json",
        "code/numerical_stability.py",
        "code/postreview_analysis.py",
        "results/audit/matching_audit_summary.json",
        "artifacts/model_selection/model_selection.csv",
        "artifacts/model_selection/table10_candidates.csv",
        "configs/postreview_exploratory.json",
        "scripts/run_all.py", "scripts/export_cell_predictions.py",
        "scripts/recompute_h1b.py",
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
    check(seed.get("numerical_stability") == 20260927, "numerical-stability seed is recorded")
    check(seed.get("reviewedv8_postreview_exploratory") == 20260927, "REVIEWEDv8 exploratory seed is recorded")

    prereg = json.loads((ROOT / "results" / "prereg_H1.json").read_text(encoding="utf-8"))
    check(prereg.get("data_split", {}).get("test") == [2021, 2023], "locked test window is 2021-2023")
    check(bool(prereg.get("k_selected")) and bool(prereg.get("rounds")), "locked model configuration present")

    selection_path = ROOT / "artifacts" / "model_selection" / "model_selection.csv"
    if selection_path.exists():
        with selection_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        check(len(rows) >= 30, f"model-selection log has {len(rows)} rows")
        check(any(row.get("selected") == "true" for row in rows), "model-selection log records selected candidates")

    table10_path = ROOT / "artifacts" / "model_selection" / "table10_candidates.csv"
    if table10_path.exists():
        with table10_path.open(encoding="utf-8-sig", newline="") as handle:
            table10 = list(csv.DictReader(handle))
        check(len(table10) == 27, "Table 10 log contains all 27 recorded fits")
        check(sum(row.get("cap_extension") == "False" for row in table10) == 25, "Table 10 log identifies 25 same-cap candidates")
        check(sum(row.get("selected_same_cap") == "True" for row in table10) == 5, "Table 10 log identifies five selected representations")
        check(sum(row.get("cap_extension") == "True" for row in table10) == 2, "Table 10 log identifies two historical cap extensions")

    ablation_path = ROOT / "results" / "supplementary" / "postreview_ablation_windows.json"
    if ablation_path.exists():
        ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
        check(ablation.get("candidate_count") == 27, "post-review ablation references the full candidate log")
        check(ablation.get("same_cap_candidate_count") == 25, "post-review ablation uses the common 4,000-round cap")
        check(set(ablation.get("metrics", {}).get("by_year", {})) == {"2021", "2022", "2023", "2024"}, "post-review ablation reports four annual windows")
        status = ablation.get("interpretation", "").lower()
        check("exploratory" in status and "cannot" in status, "post-review ablation is explicitly non-confirmatory")

    missing_path = ROOT / "results" / "supplementary" / "missing_ded_sensitivity.json"
    if missing_path.exists():
        missing = json.loads(missing_path.read_text(encoding="utf-8"))
        check(missing.get("missing_claims", {}).get("positive_claims") == 1123, "missing-deductible sensitivity includes all 1,123 positive claims")
        check(abs(missing.get("missing_claims", {}).get("amount", 0) - 74396139.94) < 0.01, "missing-deductible amount matches the audit")
        county = missing.get("scenarios", {}).get("missing_ded_county_year", {}).get("allocation_audit", {})
        check(abs(county.get("unallocated_positive_claims", -1)) < 1e-9, "county-year sensitivity allocates every missing-deductible claim")
        check(abs(county.get("unallocated_amount", -1)) < 0.01, "county-year sensitivity allocates the full missing-deductible amount")

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

    h1b_path = ROOT / "results" / "audit" / "h1b_estimands.json"
    if h1b_path.exists():
        h1b = json.loads(h1b_path.read_text(encoding="utf-8"))
        check(h1b.get("bootstrap_distances_recomputed") is True, "H1b distance bootstrap was recomputed")
        check(h1b.get("source_rows") == 399128, "H1b audit uses all test-cell predictions")
        check(h1b.get("bootstrap_unit") == "county", "H1b bootstrap unit is county")
        check(h1b.get("paired_resampling") is True, "H1b uses paired resampling")
        check(h1b.get("overall_point_rule_met") is False, "locked H1b point rule remains not met")
        if prediction_path.exists():
            check(h1b.get("source_sha256") == sha256(prediction_path), "H1b source SHA-256 matches predictions")
        expected = {
            "OE": -0.4673294105883823,
            "slope": -0.44945498603962064,
            "max_dec_dev": 10.39638169964159,
        }
        observed = {item["metric"]: item["distance_difference"] for item in h1b.get("components", [])}
        check(
            all(name in observed and abs(observed[name] - value) < 1e-12 for name, value in expected.items()),
            "H1b point estimands match the locked results",
        )
    else:
        check(False, "H1b audit result present")

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
