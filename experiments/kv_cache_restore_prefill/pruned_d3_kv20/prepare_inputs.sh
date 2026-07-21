#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ma-user/workspace/csm/mobiinfer}
EXP="$ROOT/experiments/kv_cache_restore_prefill/pruned_d3_kv20"
PROMPTS="$ROOT/artifacts/lookahead_bench/calib20k_100"
CONFIG_SRC="$ROOT/artifacts/lookahead_bench/ngram_trials/config_cpu_retest_w8g128_vocab_pruned_d3_v3_192_template.json"

mkdir -p "$EXP/logs_split" "$EXP/logs_restore" "$EXP/prefixcache" "$EXP/dump_tmp" "$EXP/runs"

perl -MJSON::PP -0777 -e '
    my ($prefix_cache) = @ARGV;
    my $data = decode_json(<STDIN>);
    $data->{use_template} = JSON::PP::false;
    $data->{prefix_cache_path} = $prefix_cache;
    print JSON::PP->new->canonical->pretty->encode($data);
' "$EXP/prefixcache" < "$CONFIG_SRC" > "$EXP/config_pruned_d3_kv.json"

for index in $(seq -w 0 19); do
    run_dir="$EXP/runs/$index"
    mkdir -p "$run_dir/tmp"

    perl -ne 'last if /^### Current Task/; print' \
        "$PROMPTS/prompt_${index}.txt" > "$run_dir/prefix_raw.txt"
    perl -ne '$emit = 1 if /^### Current Task/; print if $emit' \
        "$PROMPTS/prompt_${index}.txt" > "$run_dir/variable_raw.txt"

    {
        printf '<|im_start|>user\n'
        cat "$run_dir/prefix_raw.txt"
    } > "$run_dir/prefix.txt"
    {
        cat "$run_dir/variable_raw.txt"
        printf '<|im_end|>\n<|im_start|>assistant\n'
    } > "$run_dir/variable.txt"

    case "$index" in
        08|14|17|19) printf 'pruned_d3_prefix_b\n' > "$run_dir/cache_name.txt" ;;
        *)           printf 'pruned_d3_prefix_a\n' > "$run_dir/cache_name.txt" ;;
    esac
done

sha256sum "$EXP"/runs/*/prefix.txt > "$EXP/prefix_sha256.txt"
