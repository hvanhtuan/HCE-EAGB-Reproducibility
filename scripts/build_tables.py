"""Build machine-readable summary tables from replay or reference JSON."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "kb2_test.json"
if not SOURCE.exists():
    SOURCE = RESULTS / "reference" / "kb2_test.json"
OUTPUT = RESULTS / "tables"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten(prefix: str, value: object, row: dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), child, row)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        row[prefix] = value
    else:
        row[prefix] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    metric_rows = []
    for model, metrics in data["eval"].items():
        row: dict[str, object] = {"model": model}
        flatten("", metrics, row)
        metric_rows.append(row)
    write_csv(OUTPUT / "h1_model_metrics.csv", metric_rows)

    bootstrap_rows = []
    for comparison, values in data["boot"].items():
        bootstrap_rows.append({"comparison": comparison, **values})
    write_csv(OUTPUT / "h1_bootstrap_comparisons.csv", bootstrap_rows)

    audit = RESULTS / "audit" / "matching_audit_summary.json"
    if audit.exists():
        audit_data = json.loads(audit.read_text(encoding="utf-8"))
        row: dict[str, object] = {}
        flatten("", audit_data, row)
        write_csv(OUTPUT / "matching_audit_summary.csv", [row])

    # Preserve a tabular rendering of every committed result source. These files
    # make all reported JSON values inspectable without manual transcription,
    # while the named tables above provide the main-paper views.
    manifest_rows = []
    for namespace in ("reference", "supplementary", "audit"):
        for source in sorted((RESULTS / namespace).glob("*.json")):
            value = json.loads(source.read_text(encoding="utf-8"))
            row: dict[str, object] = {}
            flatten("", value, row)
            target = OUTPUT / "sources" / namespace / f"{source.stem}.csv"
            write_csv(target, [row])
            manifest_rows.append({
                "namespace": namespace,
                "source_json": source.relative_to(ROOT).as_posix(),
                "table_csv": target.relative_to(ROOT).as_posix(),
                "field_count": len(row),
            })
    write_csv(OUTPUT / "all_result_sources_manifest.csv", manifest_rows)

    print(f"Built tables from {SOURCE.relative_to(ROOT)} in {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
