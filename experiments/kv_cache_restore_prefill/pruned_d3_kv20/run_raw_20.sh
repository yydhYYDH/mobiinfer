#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ma-user/workspace/csm/mobiinfer}
EXP="$ROOT/experiments/kv_cache_restore_prefill/pruned_d3_kv20"
CONFIG="$ROOT/artifacts/lookahead_bench/ngram_trials/config_w8g128_vocab_pruned_d3_v3_hash_temp_20.json"
OUTPUT_DIR="$EXP/logs_raw_current"
mkdir -p "$OUTPUT_DIR"

export ROOT CONFIG OUTPUT_DIR
seq -w 0 19 | xargs -I{} -P 4 bash -lc '
    set -euo pipefail
    cd "$ROOT"
    "$ROOT/build/llm_demo" "$CONFIG" \
        "$ROOT/artifacts/lookahead_bench/calib20k_100/prompt_{}.txt" 192 \
        > "$OUTPUT_DIR/case_{}.log" 2>&1
'
