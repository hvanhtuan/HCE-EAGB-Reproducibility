#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/oracle/prepare_inputs.py"
docker build -t hce-oracle-r -f "$ROOT/oracle/Dockerfile" "$ROOT"
docker run --rm -v "$ROOT:/workspace" hce-oracle-r Rscript /workspace/oracle/oracle_metrics.R /workspace
docker run --rm -v "$ROOT:/workspace" hce-oracle-r Rscript /workspace/oracle/oracle_synthetic.R /workspace
docker run --rm -v "$ROOT:/workspace" hce-oracle-r Rscript /workspace/oracle/oracle_glm.R /workspace
python "$ROOT/oracle/finalize_oracle.py"
