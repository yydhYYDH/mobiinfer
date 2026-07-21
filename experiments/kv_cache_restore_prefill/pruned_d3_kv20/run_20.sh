#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ma-user/workspace/csm/mobiinfer}
EXP="$ROOT/experiments/kv_cache_restore_prefill/pruned_d3_kv20"
BIN="$ROOT/build/kvcache_demo"
CONFIG="$EXP/config_pruned_d3_kv.json"
export LD_LIBRARY_PATH="$ROOT/build:$ROOT/build/express:$ROOT/build/tools/cv:$ROOT/build/tools/audio"

mode=${1:?usage: run_20.sh split|restore}
case "$mode" in
    split) output_dir="$EXP/logs_split" ;;
    restore) output_dir="$EXP/logs_restore" ;;
    *) echo "unsupported mode: $mode" >&2; exit 2 ;;
esac

export ROOT EXP BIN CONFIG output_dir mode
seq -w 0 19 | xargs -I{} -P 4 bash -lc '
    set -euo pipefail
    run_dir="$EXP/runs/{}"
    cd "$run_dir"
    if [[ "$mode" == split ]]; then
        "$BIN" "$CONFIG" --split-step prefix.txt variable.txt 192 \
            > "$output_dir/case_{}.log" 2>&1
    else
        cache_name=$(cat cache_name.txt)
        "$BIN" "$CONFIG" --load-prefix-step "$cache_name" prefix.txt variable.txt 192 \
            > "$output_dir/case_{}.log" 2>&1
    fi
'
