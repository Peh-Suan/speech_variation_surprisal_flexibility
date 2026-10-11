#!/usr/bin/env bash
#
# Run the whole pipeline for one (unit, corpus, window), logging as it goes.
#
#   ./runs/run_all.sh syllable aemri            # base window
#   ./runs/run_all.sh syllable aemri prop25     # widened articulatory window
#   ./runs/run_all.sh phone beus base --rebuild-dataset
#
# Stage 1 (building the dataset) is destructive and is skipped unless you pass
# --rebuild-dataset. Everything else skips work that already exists, so
# re-running after an interruption picks up where it stopped; add --overwrite
# to force a stage to redo.
#
# Logs go to runs/logs/<unit>_<corpus>_<window>.log. Safe to run under nohup:
#
#   nohup ./runs/run_all.sh syllable aemri > /dev/null 2>&1 &
#   tail -f runs/logs/syllable_aemri_base.log

set -uo pipefail

# The STN's grid_sample has no backward kernel on MPS; without this the
# video corpora die at the first backward pass. Exported rather than left to
# the shell so it cannot go missing in a fresh terminal.
export PYTORCH_ENABLE_MPS_FALLBACK=1

UNIT="${1:?usage: run_all.sh <phone|syllable> <corpus> [window] [--rebuild-dataset] [--overwrite]}"
CORPUS="${2:?corpus required}"
WINDOW="${3:-base}"
shift 3 2>/dev/null || shift $#

REBUILD=0
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --rebuild-dataset) REBUILD=1 ;;
    *) EXTRA+=("$arg") ;;
  esac
done

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Which interpreter.
#
# `python3` is not reliable here: Homebrew's python@3.x can sit ahead of an
# active conda environment on PATH, so `python3` runs an interpreter with none
# of this project's packages while the prompt still shows the environment name.
# Prefer the active environment's own binary, and let PYTHON override.
# ---------------------------------------------------------------------------
if [ -z "${PYTHON:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
  elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
  else
    PYTHON="$(command -v "$PYTHON" || command -v python)"
  fi
fi
if [ -z "$PYTHON" ] || ! "$PYTHON" -c "import sys" 2>/dev/null; then
  echo "No usable python found. Activate your environment, or set PYTHON=/path/to/python."
  exit 1
fi

mkdir -p runs/logs
LOG="runs/logs/${UNIT}_${CORPUS}_${WINDOW}.log"

say() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

run() {
  local label="$1"; shift
  say "=== $label ==="
  if ! "$@" 2>&1 | tee -a "$LOG"; then
    say "FAILED at: $label"
    say "Fix it, then re-run this same command -- completed stages are skipped."
    exit 1
  fi
}

say "pipeline: unit=$UNIT corpus=$CORPUS window=$WINDOW"
"$PYTHON" -c "import torch; print('torch', torch.__version__,
      '| mps', torch.backends.mps.is_available())" 2>&1 | tee -a "$LOG"

if [ "$REBUILD" = "1" ]; then
  run "stage 1  dataset" \
    "$PYTHON" runs/build_dataset.py --unit "$UNIT" --corpus "$CORPUS" --overwrite
  # A new dataset invalidates the lexical columns too.
  run "stage 2  surprisal" \
    "$PYTHON" runs/compute_surprisal.py --unit "$UNIT" --corpus "$CORPUS" --overwrite
else
  run "stage 2  surprisal" \
    "$PYTHON" runs/compute_surprisal.py --unit "$UNIT" --corpus "$CORPUS" "${EXTRA[@]+"${EXTRA[@]}"}"
fi

# The acoustic encoder does not depend on the window, so a widened run trains
# only the articulatory one and reuses the acoustic checkpoint.
ONLY=()
[ "$WINDOW" != "base" ] && ONLY=(--only art)

run "stage 3  encoders" \
  "$PYTHON" runs/train_encoders.py --unit "$UNIT" --corpus "$CORPUS" --window "$WINDOW" \
          "${ONLY[@]+"${ONLY[@]}"}" "${EXTRA[@]+"${EXTRA[@]}"}"
run "stage 4  latents" \
  "$PYTHON" runs/extract_latents.py --unit "$UNIT" --corpus "$CORPUS" --window "$WINDOW" "${EXTRA[@]+"${EXTRA[@]}"}"
run "stage 5  distances" \
  "$PYTHON" runs/compute_distances.py --unit "$UNIT" --corpus "$CORPUS" --window "$WINDOW" "${EXTRA[@]+"${EXTRA[@]}"}"
run "stage 6  entropy" \
  "$PYTHON" runs/estimate_entropy.py --unit "$UNIT" --corpus "$CORPUS" --window "$WINDOW" "${EXTRA[@]+"${EXTRA[@]}"}"
run "stage 7  mutual information" \
  "$PYTHON" runs/estimate_mi.py --unit "$UNIT" --corpus "$CORPUS" --window "$WINDOW" "${EXTRA[@]+"${EXTRA[@]}"}"
run "stage 8  combine" \
  "$PYTHON" runs/combine.py --unit "$UNIT" --corpus "$CORPUS" --window "$WINDOW" "${EXTRA[@]+"${EXTRA[@]}"}"

say "done: unit=$UNIT corpus=$CORPUS window=$WINDOW"
