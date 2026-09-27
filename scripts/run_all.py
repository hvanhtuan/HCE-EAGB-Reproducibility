"""Cross-platform, logged entry point for release verification and full replay."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
LOG_DIR = ROOT / "artifacts" / "logs"


def run(command: list[str], cwd: Path, log_handle) -> None:
    printable = " ".join(command)
    print(f"\n>>> {printable}")
    log_handle.write(f"\n>>> {printable}\n")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)
    return_code = process.wait()
    if return_code:
        raise SystemExit(f"Command failed with exit code {return_code}: {printable}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("verify", "full"), default="verify")
    parser.add_argument("--rebuild-cells", action="store_true")
    parser.add_argument("--skip-fremtpl2", action="store_true")
    parser.add_argument(
        "--include-postreview-exploratory",
        action="store_true",
        help="also rerun the approximately one-hour REVIEWEDv8 ablation and missing-deductible sensitivity analysis",
    )
    args = parser.parse_args()

    seeds = json.loads((ROOT / "configs" / "seeds.json").read_text(encoding="utf-8"))
    seed = str(seeds["global"])
    os.environ.update({
        "PYTHONHASHSEED": seed,
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "MPLBACKEND": "Agg",
    })

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"run_{args.mode}_{stamp}.log"
    started = dt.datetime.now(dt.timezone.utc)

    with log_path.open("w", encoding="utf-8") as log:
        if args.mode == "verify":
            run([sys.executable, "scripts/verify_release.py"], ROOT, log)
        else:
            cells = ROOT / "data" / "nfip_cells.parquet"
            if args.rebuild_cells or not cells.exists():
                run([sys.executable, "prep_nfip.py"], CODE, log)
            run([sys.executable, "kb2_h1.py", "test"], CODE, log)
            run([sys.executable, "kb3_robust.py"], CODE, log)
            run([sys.executable, "spatial_oos.py"], CODE, log)
            if not args.skip_fremtpl2:
                run([sys.executable, "kb1_fremtpl2.py"], CODE, log)
                run([sys.executable, "kb1_paired.py"], CODE, log)
            run([sys.executable, "scripts/export_cell_predictions.py"], ROOT, log)
            run([sys.executable, "scripts/recompute_h1b.py"], ROOT, log)
            run([sys.executable, "scripts/build_model_selection_log.py"], ROOT, log)
            run([sys.executable, "scripts/build_tables.py"], ROOT, log)
            run([sys.executable, "build_all.py"], ROOT / "figures", log)
            if args.include_postreview_exploratory:
                run([sys.executable, "postreview_analysis.py"], CODE, log)
            run([sys.executable, "scripts/generate_checksums.py"], ROOT, log)
            run([sys.executable, "scripts/verify_release.py"], ROOT, log)

    manifest = {
        "release": "1.0.0",
        "mode": args.mode,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "seed": int(seed),
        "thread_limit": 2,
        "log": str(log_path.relative_to(ROOT)),
    }
    manifest_path = LOG_DIR / f"run_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCompleted. Log: {log_path}")
    print(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
