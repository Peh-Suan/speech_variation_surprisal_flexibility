#!/usr/bin/env bash
#
# Run several analyses one after another, unattended.
#
# MPS is a single device, so these do NOT overlap: two concurrent training jobs
# split one GPU rather than using two, and the bigger models can exhaust
# unified memory. Serial with a queue is both faster overall and safer.
#
#   ./runs/queue.sh syllable base aemri75 aemri75anno aeema frmri beus
#   ./runs/queue.sh syllable prop25 aemri aemri75 frmri
#
# Builds the feature cache first when one is missing, then runs stages 2-8.
# A corpus that fails does not stop the rest -- the summary at the end says
# which succeeded, and re-running the same command resumes each from where it
# stopped.

set -uo pipefail

# The STN's grid_sample has no backward kernel on MPS; without this the
# video corpora die at the first backward pass. Exported rather than left to
# the shell so it cannot go missing in a fresh terminal.
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$(dirname "$0")/.."

if [ -z "${PYTHON:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
  elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
  else
    PYTHON="$(command -v python3 || command -v python)"
  fi
fi

UNIT="${1:?usage: queue.sh <phone|syllable> <window> <corpus> [corpus...]}"
WINDOW="${2:?window required, e.g. base or prop25}"
shift 2
[ "$#" -gt 0 ] || { echo "name at least one corpus"; exit 1; }

mkdir -p runs/logs
SUMMARY="runs/logs/queue_${UNIT}_${WINDOW}.summary"
: > "$SUMMARY"

started=$(date +%s)
for corpus in "$@"; do
  echo
  echo "=============================================================="
  echo " $(date +%H:%M:%S)  $UNIT / $corpus / $WINDOW"
  echo "=============================================================="
  one=$(date +%s)

  "$PYTHON" runs/build_cache.py --unit "$UNIT" --corpus "$corpus" --window "$WINDOW" \
      --verify 0 || echo "  (cache build skipped or failed; continuing without it)"

  if ./runs/run_all.sh "$UNIT" "$corpus" "$WINDOW"; then
    status="ok"
  else
    status="FAILED"
  fi
  printf '%-14s %-8s %-8s %s  (%d min)\n' "$corpus" "$UNIT" "$WINDOW" "$status" \
      $(( ($(date +%s) - one) / 60 )) | tee -a "$SUMMARY"
done

echo
echo "=============================================================="
echo " queue finished in $(( ($(date +%s) - started) / 60 )) min"
cat "$SUMMARY"
echo "=============================================================="
