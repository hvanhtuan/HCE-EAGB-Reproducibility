$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
python (Join-Path $PSScriptRoot 'prepare_inputs.py')
docker build -t hce-oracle-r -f (Join-Path $PSScriptRoot 'Dockerfile') $root
docker run --rm -v "${root}:/workspace" hce-oracle-r Rscript /workspace/oracle/oracle_metrics.R /workspace
docker run --rm -v "${root}:/workspace" hce-oracle-r Rscript /workspace/oracle/oracle_synthetic.R /workspace
docker run --rm -v "${root}:/workspace" hce-oracle-r Rscript /workspace/oracle/oracle_glm.R /workspace
python (Join-Path $PSScriptRoot 'finalize_oracle.py')
