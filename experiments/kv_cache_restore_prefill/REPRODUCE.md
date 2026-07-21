# 复现命令

## 服务器快速测试

```bash
ssh sai-a100
cd /home/ma-user/workspace/csm/mobiinfer
eval "$(conda shell.bash hook)"
conda activate mnn

mkdir -p experiments/kv_cache_restore_prefill/{inputs,logs,tmp}
```

如果需要应用本次修复补丁：

```bash
git apply experiments/kv_cache_restore_prefill/kv_cache_restore_tmpdir_fix.patch
cmake --build build --target kvcache_demo -j8
```

运行顺序：

```bash
LD_LIBRARY_PATH="$PWD/build:$PWD/build/express:$PWD/build/tools/cv:$PWD/build/tools/audio" \
  "$PWD/build/kvcache_demo" \
  "$PWD/transformers/llm/export/model/config.json" \
  --split-step \
  "$PWD/experiments/kv_cache_restore_prefill/inputs/prefix.txt" \
  "$PWD/experiments/kv_cache_restore_prefill/inputs/variable.txt" \
  1 \
  > "$PWD/experiments/kv_cache_restore_prefill/logs/server_split_step.log" 2>&1

LD_LIBRARY_PATH="$PWD/build:$PWD/build/express:$PWD/build/tools/cv:$PWD/build/tools/audio" \
  "$PWD/build/kvcache_demo" \
  "$PWD/transformers/llm/export/model/config.json" \
  --dump \
  server_prompt00_prefix \
  "$PWD/experiments/kv_cache_restore_prefill/inputs/prefix.txt" \
  > "$PWD/experiments/kv_cache_restore_prefill/logs/server_dump.log" 2>&1

LD_LIBRARY_PATH="$PWD/build:$PWD/build/express:$PWD/build/tools/cv:$PWD/build/tools/audio" \
  "$PWD/build/kvcache_demo" \
  "$PWD/transformers/llm/export/model/config.json" \
  --load-prefix-step \
  server_prompt00_prefix \
  "$PWD/experiments/kv_cache_restore_prefill/inputs/prefix.txt" \
  "$PWD/experiments/kv_cache_restore_prefill/inputs/variable.txt" \
  1 \
  > "$PWD/experiments/kv_cache_restore_prefill/logs/server_load_prefix_step.log" 2>&1
```

## 手机正式 case

```bash
ssh oneplus13
cd /data/data/com.termux/files/home/csm/mobiinfer

EXP="$PWD/experiments/kv_cache_restore_prefill"
mkdir -p "$EXP"/{inputs,logs,tmp,prefixcache}
```

运行顺序：

```bash
LD_LIBRARY_PATH="$EXP/bin:$PWD" \
  "$EXP/bin/kvcache_demo" \
  "$PWD/bench20/configs/config_baseline_w8a8_ar.json" \
  --split-step \
  "$EXP/inputs/prefix.txt" \
  "$EXP/inputs/variable.txt" \
  1 \
  > "$EXP/logs/phone_split_step.log" 2>&1

LD_LIBRARY_PATH="$EXP/bin:$PWD" \
  "$EXP/bin/kvcache_demo" \
  "$PWD/bench20/configs/config_baseline_w8a8_ar.json" \
  --dump \
  phone_prompt00_prefix \
  "$EXP/inputs/prefix.txt" \
  > "$EXP/logs/phone_dump.log" 2>&1

LD_LIBRARY_PATH="$EXP/bin:$PWD" \
  "$EXP/bin/kvcache_demo" \
  "$PWD/bench20/configs/config_baseline_w8a8_ar.json" \
  --load-prefix-step \
  phone_prompt00_prefix \
  "$EXP/inputs/prefix.txt" \
  "$EXP/inputs/variable.txt" \
  1 \
  > "$EXP/logs/phone_load_prefix_step.log" 2>&1
```

## 结果提取

```bash
grep -E "tokens num|prefill time|decode time|KV cache dump done" experiments/kv_cache_restore_prefill/logs/*.log
```

## 服务器图片一致性验证

以下命令在模型目录执行，以保证相对的 `tmp` 和 `prefixcache` 路径一致：

```bash
cd /home/ma-user/workspace/csm/mobiinfer/transformers/llm/export/model
ROOT=/home/ma-user/workspace/csm/mobiinfer
EXP="$ROOT/experiments/kv_cache_restore_prefill"
export LD_LIBRARY_PATH="$ROOT/build:$ROOT/build/express:$ROOT/build/tools/cv:$ROOT/build/tools/audio"

"$ROOT/build/kvcache_demo" config.greedy.json \
  --split-step "$EXP/inputs/prefix_img.txt" "$EXP/inputs/variable_img.txt" 32 \
  > "$EXP/logs/final_img_split_greedy32.log" 2>&1

"$ROOT/build/kvcache_demo" config.greedy.json \
  --dump check_img_prefix "$EXP/inputs/prefix_img.txt" \
  > "$EXP/logs/final_img_dump.log" 2>&1

"$ROOT/build/kvcache_demo" config.greedy.json \
  --load-prefix-step check_img_prefix "$EXP/inputs/prefix_img.txt" "$EXP/inputs/variable_img.txt" 32 \
  > "$EXP/logs/final_img_load_greedy32.log" 2>&1
```

清洗诊断行并逐字节比较：

```bash
strip_answer() {
  awk '/#################################/{exit} \
       !/^config path is / && !/^CPU Group:/ && \
       !/^The device supports:/ && !/^Loaded prefix/{print}' "$1"
}

strip_answer "$EXP/logs/final_img_split_greedy32.log" > /tmp/final_img_split.answer
strip_answer "$EXP/logs/final_img_load_greedy32.log" > /tmp/final_img_load.answer
cmp /tmp/final_img_split.answer /tmp/final_img_load.answer
sha256sum /tmp/final_img_split.answer /tmp/final_img_load.answer
```

## A8W8 Lookahead 与 Prefix Restore 组合

配置和模板化拆分输入已放在：

```text
experiments/kv_cache_restore_prefill/lookahead_cpu/
```

运行：

```bash
cd /home/ma-user/workspace/csm/mobiinfer/experiments/kv_cache_restore_prefill/lookahead_cpu
ROOT=/home/ma-user/workspace/csm/mobiinfer
export LD_LIBRARY_PATH="$ROOT/build:$ROOT/build/express:$ROOT/build/tools/cv:$ROOT/build/tools/audio"

"$ROOT/build/kvcache_demo" config_ar.json \
  --dump a8w8_prompt00_template_prefix prefix_template.txt \
  > logs/dump_prefix.log 2>&1

"$ROOT/build/kvcache_demo" config_ar.json \
  --split-step prefix_template.txt variable_template.txt 192 \
  > logs/ar_split_192.log 2>&1

"$ROOT/build/kvcache_demo" config_ar.json \
  --load-prefix-step a8w8_prompt00_template_prefix prefix_template.txt variable_template.txt 192 \
  > logs/ar_restore_192.log 2>&1

"$ROOT/build/kvcache_demo" config_lookahead.json \
  --split-step prefix_template.txt variable_template.txt 192 \
  > logs/lookahead_split_192.log 2>&1

"$ROOT/build/kvcache_demo" config_lookahead.json \
  --load-prefix-step a8w8_prompt00_template_prefix prefix_template.txt variable_template.txt 192 \
  > logs/lookahead_restore_192.log 2>&1
```

关键字段提取：

```bash
grep -E "Loaded .*ngram|prefix tokens num|variable tokens num|decode tokens num|prefill time|decode time" logs/*.log
```

## Pruned Lookahead D3 + KV Restore，20 Cases

```bash
cd /home/ma-user/workspace/csm/mobiinfer/experiments/kv_cache_restore_prefill/pruned_d3_kv20

# 生成配置、两类模板化 prefix、20 个 continuation 和独立 tmp 目录
./prepare_inputs.sh

# 分别从 A/B 代表 case dump 两份 prefix cache
cd dump_tmp
export LD_LIBRARY_PATH=/home/ma-user/workspace/csm/mobiinfer/build:/home/ma-user/workspace/csm/mobiinfer/build/express:/home/ma-user/workspace/csm/mobiinfer/build/tools/cv:/home/ma-user/workspace/csm/mobiinfer/build/tools/audio
/home/ma-user/workspace/csm/mobiinfer/build/kvcache_demo ../config_pruned_d3_kv.json \
  --dump pruned_d3_prefix_a ../runs/00/prefix.txt > ../dump_prefix_a.log 2>&1
/home/ma-user/workspace/csm/mobiinfer/build/kvcache_demo ../config_pruned_d3_kv.json \
  --dump pruned_d3_prefix_b ../runs/08/prefix.txt > ../dump_prefix_b.log 2>&1

# 按 README 的 4 路并行分别运行 split 和 restore
cd ..
./run_20.sh split
./run_20.sh restore

# 逐 case 哈希比较并生成 results.json/results.csv
./summarize_results.pl . > summary.txt

# 当前二进制 raw full 对照及 raw/split 全文比较
./run_raw_20.sh
./compare_raw_split.pl . > raw_split_comparison.stdout
```

日志位置：

```text
pruned_d3_kv20/logs_split/case_00.log ... case_19.log
pruned_d3_kv20/logs_restore/case_00.log ... case_19.log
```
