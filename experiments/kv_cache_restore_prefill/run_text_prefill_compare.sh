#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ma-user/workspace/csm/mobiinfer"
EXP="$ROOT/experiments/kv_cache_restore_prefill"
MODEL="$ROOT/transformers/llm/export/model"
CONFIG="${1:-config.greedy.json}"
MAX_TOKENS="${2:-32}"

mkdir -p "$EXP/inputs" "$EXP/logs"

PREFIX_FILE="${PREFIX_FILE:-$EXP/inputs/prefix.txt}"
VARIABLE_FILE="${VARIABLE_FILE:-$EXP/inputs/variable.txt}"
FULL_FILE="${FULL_FILE:-$EXP/inputs/text_full.txt}"

if [[ ! -f "$PREFIX_FILE" || ! -f "$VARIABLE_FILE" ]]; then
  echo "Missing input files:"
  echo "  PREFIX_FILE=$PREFIX_FILE"
  echo "  VARIABLE_FILE=$VARIABLE_FILE"
  exit 1
fi

cat "$PREFIX_FILE" "$VARIABLE_FILE" > "$FULL_FILE"

cd "$MODEL"
rm -rf tmp prefixcache

export LD_LIBRARY_PATH="$ROOT/build:$ROOT/build/express:$ROOT/build/tools/cv:$ROOT/build/tools/audio:${LD_LIBRARY_PATH:-}"

echo "[1/2] split-step: prefix prefill + variable prefill"
"$ROOT/build/kvcache_demo" \
  "$CONFIG" \
  --split-step \
  "$PREFIX_FILE" \
  "$VARIABLE_FILE" \
  "$MAX_TOKENS" \
  > "$EXP/logs/text_split_step_${CONFIG%.json}_${MAX_TOKENS}.log" 2>&1

echo "[2/2] raw: full prompt prefill"
"$ROOT/build/kvcache_demo" \
  "$CONFIG" \
  --raw \
  "$FULL_FILE" \
  "$MAX_TOKENS" \
  > "$EXP/logs/text_raw_full_${CONFIG%.json}_${MAX_TOKENS}.log" 2>&1

echo
echo "Logs:"
echo "  $EXP/logs/text_split_step_${CONFIG%.json}_${MAX_TOKENS}.log"
echo "  $EXP/logs/text_raw_full_${CONFIG%.json}_${MAX_TOKENS}.log"
echo
grep -E "tokens num|prefill time|decode time|################################" \
  "$EXP/logs/text_split_step_${CONFIG%.json}_${MAX_TOKENS}.log" \
  "$EXP/logs/text_raw_full_${CONFIG%.json}_${MAX_TOKENS}.log"
