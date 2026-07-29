#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT=${ROOT:-/data/data/com.termux/files/home/csm/mobiinfer}
BENCH=${BENCH:-$ROOT/multiturn_bench}
MAX_NEW=${MAX_NEW:-192}

BASELINE_CONFIG=${BASELINE_CONFIG:-$ROOT/bench20/configs/config_baseline_w8a8_ar.json}
MOBIINFER_CONFIG=${MOBIINFER_CONFIG:-$ROOT/bench20/configs/config_vocab_pruned_w8a8_la_d3_bin.json}

BASELINE_LOGS="$BENCH/logs_raw_baseline"
MOBIINFER_LOGS="$BENCH/logs_raw_mobiinfer"
mkdir -p "$BASELINE_LOGS" "$MOBIINFER_LOGS"

run_one() {
  local name=$1
  local config=$2
  local prompt=$3
  local log=$4
  if grep -q "#################################" "$log" 2>/dev/null; then
    echo "skip $name"
    return
  fi
  echo "run $name"
  LD_LIBRARY_PATH="$ROOT" "$ROOT/llm_demo" "$config" "$prompt" "$MAX_NEW" > "$log" 2>&1
}

tail -n +2 "$BENCH/manifest.tsv" | while IFS=$'\t' read -r traj step history prompt image task assistant; do
  case_name="traj_${traj}_step_${step}"
  prompt_path="$BENCH/$prompt"
  test -f "$prompt_path"
  run_one "baseline_${case_name}" \
    "$BASELINE_CONFIG" \
    "$prompt_path" \
    "$BASELINE_LOGS/${case_name}.log"
  run_one "mobiinfer_${case_name}" \
    "$MOBIINFER_CONFIG" \
    "$prompt_path" \
    "$MOBIINFER_LOGS/${case_name}.log"
done

python "$BENCH/summarize_multiturn_logs.py" --bench "$BENCH"

