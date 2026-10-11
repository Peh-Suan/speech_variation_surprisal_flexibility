#!/usr/bin/env bash
#
# Finish BEUS phone at the base window.
#
# Stages 1-3 are already done: the dataset and surprisal were rebuilt with the
# British English dictionary, and the phone encoders are unchanged (the rebuild
# altered 2.4% of phone LABELS but not a single token boundary, and encoders
# never see labels).
#
# Stages 4 and 5 are redone with --overwrite because the earlier run required
# both streams per token, which cost ~3,500 articulatory-only tokens.
#
#     nohup ./runs/finish_beus.sh > /dev/null 2>&1 &
#     tail -f runs/logs/finish_beus.log

set -uo pipefail
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

mkdir -p runs/logs
LOG=runs/logs/finish_beus.log
: > "$LOG"

if [ ! -f phone_based/encoder_training/ckpts/beus/art_model_ckpt.pth ]; then
  echo "Missing the BEUS phone encoders. Run ./runs/run_all.sh phone beus base instead."
  exit 1
fi

run() {
  printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$LOG"
  shift
  "$@" 2>&1 | tee -a "$LOG" || { echo "FAILED — fix, then re-run this script" | tee -a "$LOG"; exit 1; }
}

run "stage 4  latents (redo: both streams now kept independently)" \
    "$PYTHON" runs/extract_latents.py   --unit phone --corpus beus --overwrite
run "stage 5  distances (redo)" \
    "$PYTHON" runs/compute_distances.py --unit phone --corpus beus --overwrite
run "stage 6  conditional entropy" \
    "$PYTHON" runs/estimate_entropy.py  --unit phone --corpus beus
run "stage 7  mutual information" \
    "$PYTHON" runs/estimate_mi.py       --unit phone --corpus beus
run "stage 8  combine" \
    "$PYTHON" runs/combine.py           --unit phone --corpus beus

printf '\n[%s] BEUS phone/base complete — the phone analysis is now finished for all six corpora.\n' \
    "$(date +%H:%M:%S)" | tee -a "$LOG"
