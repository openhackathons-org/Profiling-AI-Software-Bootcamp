#!/usr/bin/env bash
# Run all code-bearing notebooks via nbconvert, capture results.
# Designed to be executed INSIDE the docker container at /workspace.
set -u

OUTDIR=/tmp/nbout
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.tsv"
: > "$SUMMARY"

NOTEBOOKS=(
  jupyter_notebook/system-topology.ipynb
  jupyter_notebook/data-parallelism.ipynb
  jupyter_notebook/model-parallelism.ipynb
  jupyter_notebook/multinode.ipynb
  jupyter_notebook/nsight_advanced.ipynb
  jupyter_notebook/nsys-application.ipynb
  jupyter_notebook/nsys-fp8.ipynb
  jupyter_notebook/transEng.ipynb
  jupyter_notebook/advanced_optimizations.ipynb
)

cd /workspace

for nb in "${NOTEBOOKS[@]}"; do
  name=$(basename "$nb" .ipynb)
  log="$OUTDIR/${name}.log"
  echo "=== START $nb $(date -Iseconds) ===" | tee -a "$log"
  start=$(date +%s)
  # Run from the notebook's own directory so relative paths (cd ../source_code) work.
  jupyter nbconvert --to notebook --execute "$nb" \
      --output-dir "$OUTDIR" \
      --output "${name}.ipynb" \
      --ExecutePreprocessor.timeout=1800 \
      --ExecutePreprocessor.kernel_name=python3 \
      >>"$log" 2>&1
  rc=$?
  end=$(date +%s)
  elapsed=$((end-start))
  if [ $rc -eq 0 ]; then status=PASS; else status=FAIL; fi
  printf "%s\t%s\t%ds\trc=%d\n" "$status" "$nb" "$elapsed" "$rc" | tee -a "$SUMMARY"
  echo "=== END $nb $status rc=$rc elapsed=${elapsed}s ===" | tee -a "$log"
done

echo "===== SUMMARY ====="
cat "$SUMMARY"
