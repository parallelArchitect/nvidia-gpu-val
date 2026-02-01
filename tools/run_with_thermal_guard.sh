#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX=${GPU_INDEX:-0}
MIN_START=42
MAX_START=51
COOLDOWN_TARGET=49
SLEEP_SEC=5

get_temp() {
  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i "$GPU_INDEX"
}

echo "[thermal] Waiting for GPU $GPU_INDEX to be within ${MIN_START}-${MAX_START}C..."

while true; do
  T=$(get_temp)
  echo "[thermal] temp=${T}C"
  if [ "$T" -ge "$MIN_START" ] && [ "$T" -le "$MAX_START" ]; then
    break
  fi
  sleep "$SLEEP_SEC"
done

echo "[thermal] OK to start validation"
PYTHONPATH=. python3 run_validation.py "$@"

echo "[thermal] Validation finished, cooling down to <= ${COOLDOWN_TARGET}C..."

while true; do
  T=$(get_temp)
  echo "[thermal] cooldown temp=${T}C"
  if [ "$T" -le "$COOLDOWN_TARGET" ]; then
    break
  fi
  sleep "$SLEEP_SEC"
done

echo "[thermal] Cooldown complete"
