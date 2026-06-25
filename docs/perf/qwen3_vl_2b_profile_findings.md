# Qwen3-VL-2B MNN Profile Findings

Date: 2026-06-24

Input profile:

- `logs/qwen3_vl_img_op_profile_phase.csv`
- `logs/qwen3_vl_img_op_profile_phase.log`

Model:

- `/mnt/e/WAIC/pc_server/models/Qwen3-VL-2B-Instruct-MNN/config.json`

Run type:

- Image + text prompt
- CPU x86 optimized build
- Short decode run

## Profile Columns

The CSV is split by op name and phase where possible.

Important columns:

- `phase`: `prefill`, `decode`, or `unknown`
- `avg_ms`: accumulated op time for this row
- `dispatch_ms`: `Execution::onExecute()` wall time
- `wait_ms`: output wait time in callback
- `io_bytes`: input + output tensor bytes
- `static_bytes`: registered static packed weight bytes
- `act_bw_gbps`: effective bandwidth from input/output bytes
- `total_bw_gbps`: effective bandwidth from input/output + static weight bytes
- `gflops`: effective throughput from MNN flops estimate
- `gemm_m/gemm_k/gemm_n`: GEMM shape for Linear/Convolution-as-GEMM
- `weight_bits`: selected weight bit width
- `kernel`: selected GEMM kernel
- `repack_via_int8`: whether 4-bit weight load used a temporary int8 reorder path
- `online_repack_for_prefill`: whether runtime online weight repack was used for prefill

Bandwidth values are effective estimates from profiler-visible bytes, not hardware counter measurements.

## Top-Level Bottlenecks

Total profiled op time:

```text
4107.58 ms
```

Phase summary:

```text
prefill: 2568.69 ms, 62.54%
decode:   812.20 ms, 19.77%
unknown:  726.70 ms, 17.69%
```

Op type summary:

```text
Convolution: 3380.89 ms, 82.31%
Attention:    382.10 ms,  9.30%
Raster:       143.24 ms,  3.49%
BinaryOp:      96.69 ms,  2.35%
While:         44.23 ms,  1.08%
UnaryOp:       37.37 ms,  0.91%
LayerNorm:     22.46 ms,  0.55%
```

In this LLM/VLM path, most `Convolution` rows are actually Linear/GEMM executions after MNN lowering.

## Main Findings

### 1. Prefill Linear/GEMM Is The Largest Cost

Prefill `Convolution` accounts for:

```text
2568.69 ms, 62.54% total
```

Major prefill groups:

```text
text MLP:            1179.09 ms, 28.71%
vision block matmul:  802.17 ms, 19.53%
text attention proj:  432.21 ms, 10.52%
vision deepstack:      55.11 ms,  1.34%
```

The text MLP rows are mostly W4 GEMM:

```text
kernel=Int8GemmKernel_W4
weight_bits=4
repack_via_int8=false
```

Typical text MLP shapes:

```text
gate/up:  MKN=189x2048x6144
down:     MKN=189x6144x2048
```

### 2. Decode `lm_head` Is The Largest Single Decode Op

`lm_head` appears only in decode in this profile:

```text
/lm/lm_head/Linear
phase=decode
avg_ms=197.464
calls=24
MKN=1x2048x151936
weight_bits=8
kernel=Int8GemmKernel
```

Share:

```text
4.81% of total profiled time
24.31% of decode Convolution time
```

There is no prefill `lm_head` row. The engine appears to skip full prompt logits during prefill and only runs vocab projection for decode sampling. This is expected for efficient generation.

The important issue is that `lm_head` is using 8-bit weight path, not W4:

```text
weight_bits=8
kernel=Int8GemmKernel
```

This is a clear optimization target because the vocab dimension is large.

### 3. Decode M=1 GEMM Is Also Significant

Decode `Convolution` breakdown:

```text
lm_head:       197.46 ms
mlp/gate_proj: 154.70 ms
mlp/down_proj: 149.37 ms
mlp/up_proj:   144.14 ms
attn/q_proj:    55.15 ms
attn/o_proj:    54.85 ms
attn/k_proj:    28.44 ms
attn/v_proj:    28.10 ms
```

Total decode MLP projections:

```text
448.20 ms
```

Typical decode MLP shape:

```text
MKN=1x2048x6144
MKN=1x6144x2048
```

These are W4 kernels, but M=1 has much lower effective throughput than prefill. Decode optimization should focus on small-M GEMM kernels and scheduling overhead.

### 4. Vision Block Matmul Is A Major Image-Side Cost

`vision block matmul` is an analysis grouping, not an MNN op type.

It refers to visual transformer Linear/GEMM rows such as:

```text
/Add_*_matmul_converted
/mlp/linear_fc1_*/Add_output_0__matmul_converted
/mlp/linear_fc2_*/Add_output_0__matmul_converted
```

Typical visual GEMM shapes:

```text
/Add_*:           MKN=676x1024x1024
/mlp/linear_fc1:  MKN=676x1024x4096
/mlp/linear_fc2:  MKN=676x4096x1024
```

The `M=676` likely corresponds to the visual token grid, for example a 26x26 patch/grid.

Total grouped time:

```text
802.17 ms, 19.53% total
```

These rows generally use:

```text
weight_bits=4
kernel=Int8GemmKernel_W4
repack_via_int8=false
```

So the main issue is volume of visual GEMMs rather than missing W4.

### 5. Raster Is Not The Primary Bottleneck

Raster total:

```text
143.24 ms, 3.49%
```

Main Raster groups:

```text
vision/block_matmul Raster: 62.63 ms
raster/other:              56.96 ms
text MLP Raster:            8.36 ms
text attention Raster:      6.75 ms
lm_head Raster:             1.94 ms
```

`Raster` represents tensor layout/view materialization, copies, reshapes, transposes, slicing, concat/gather-style movement, or layout adaptation around Linear.

Raster can be optimized later, but it is not the first priority compared with GEMM.

### 6. 4-Bit Repack Is Not A Runtime Bottleneck

Only two rows report:

```text
repack_via_int8=true
```

They are patch embedding rows:

```text
/patch_embed/proj/Conv_output_0__0  3.604 ms
/patch_embed/proj/Conv_output_0__1  3.481 ms
```

Both are:

```text
kernel=256
weight_bits=4
```

This path temporarily expands to int8 for weight reorder, then packs back to 4-bit. It is a load/reorder behavior, not a permanent 8-bit weight path. Total cost is about 7.1 ms, so it is not a main bottleneck.

### 7. Layer 16 Looks Suspicious

Layer 16 prefill is slower than most layers:

```text
layer 16 prefill total: 83.36 ms
most other layers:     53-63 ms
```

Top layer 16 rows:

```text
/layers.16/mlp/down_proj/Linear 22.83 ms
/layers.16/mlp/gate_proj/Linear 21.42 ms
/layers.16/mlp/up_proj/Linear   20.45 ms
```

This may be run-to-run noise, cache effects, thread scheduling, or a real layer-specific issue. It should be checked by running multiple profiles and comparing layer totals.

## Eagle Speculative Decode Notes

Additional x86 runs were captured on 2026-06-24 with:

```text
Eagle model:
/mnt/e/WAIC/pc_server/models/Qwen3-VL-2B-Instruct-Eagle3-MNN/config.json

Non-Eagle model:
/mnt/e/WAIC/pc_server/models/Qwen3-VL-2B-Instruct-MNN/config.json
```

Both used image input and a short generation length.

### Timeline Phases

The op timeline now separates:

```text
vision_prefill
prefill
decode
eagle
```

`eagle` is set by an explicit thread-local profile phase around Eagle draft
forward paths, not by op-name matching. Target-model verification remains in
`decode`.

For the Eagle run with detailed timing:

```text
vision_prefill  wall 1598.45 ms
prefill         wall 1348.46 ms
first decode    wall   11.07 ms
eagle run1      wall   73.96 ms
verify decode1  wall   57.18 ms
eagle run2      wall    9.96 ms
verify decode2  wall   59.04 ms
```

### Eagle Draft Cost

The second Eagle run is the steady-state small-M draft case:

```text
eagle run2 wall:  9.96 ms
eagle run2 op:    7.30 ms
```

Breakdown from `[MNN_PROFILE_EAGLE]` timing:

```text
eagle_forward        10.476 ms, count=4, avg=2.619 ms
embedding             2.267 ms, count=4, avg=0.567 ms
topk_read             0.645 ms, count=6, avg=0.107 ms
mask_build            0.017 ms
token_tree_grow       0.013 ms
token_tree_finalize   0.012 ms
token_tree_init       0.003 ms
```

`tokenTree` and mask construction are not the major cost in this run. The
visible draft-side cost is mostly Eagle forward plus embedding. `TopKV2` itself
is also small; earlier wall gaps were mostly synchronization / CPU preparation
around the next draft step.

### Draft Tree Size And Accept Logging

With default config values:

```text
eagle_depth = 3
eagle_topk = 1
draft_predict_length = 3
```

The generated tree is effectively a chain. The log confirms:

```text
[MNN_PROFILE_EAGLE] stage=token_tree_finalize iter=3 seqLen=181 ids=4
[MNN_PROFILE_EAGLE] stage=token_tree_finalize iter=3 seqLen=182 ids=4
```

So each Eagle draft produces 4 tree nodes for target verification.

The current release run does not print accepted-token counts. The code records
them in `accpetLens`, but printing is guarded by `EAGLE_DEBUG`:

```text
#define EAGLE_DEBUG 0
```

Turning `EAGLE_DEBUG` on will print accept/compression information, but it also
prints draft tokens, tree paths, masks, and token strings. For performance
profiling, prefer a lightweight profile log such as:

```text
[MNN_PROFILE_EAGLE] stage=accept step=<n> draft_tokens=<n> accepted=<n> total_new=<n>
```

### Target Verify Cost

Eagle target verification is a tree-structured target forward. It is similar to
a small prefill over the draft tree nodes, not a single-token decode.

Verify runs:

```text
verify decode1 wall: 57.18 ms
verify decode1 op:   51.30 ms
verify decode2 wall: 59.04 ms
verify decode2 op:   56.91 ms
```

Main GEMM shapes in verify:

```text
M=4 K=2048 N=2048   count=56
M=4 K=2048 N=1024   count=56
M=4 K=2048 N=6144   count=56
M=4 K=6144 N=2048   count=28
```

This means the target model runs all transformer layers with `M=4` tree nodes.

`lm_head` inside verify:

```text
/lm/lm_head/Linear
M=1 K=2048 N=151936
verify decode1 lm_head: ~10.22 ms
```

Verify decode1 breakdown:

```text
wall_ms: 57.18
op_ms:   51.30
lm_head: 10.22
other:   41.08
```

So target verification is dominated by the target transformer forward, not by
`lm_head`.

### Non-Eagle Decode Baseline

The non-Eagle run with 4 generated tokens produced:

```text
decode wall: 233.34 ms
decode op:   213.39 ms
lm_head:      55.12 ms
other:       158.27 ms
```

There are 5 `lm_head` calls in the decode segment: one after prefill plus four
generated-token steps.

```text
/lm/lm_head/Linear count=5
total 54.70 ms
avg   10.94 ms/call
max   12.02 ms
```

Approximate per-step baseline:

```text
wall per 5 lm_head steps: 233.34 / 5 = 46.7 ms
lm_head per step:          55.12 / 5 = 11.0 ms
other per step:           158.27 / 5 = 31.7 ms
```

This is close to the Eagle verify cost per verify step. Therefore Eagle only
wins if each verify accepts enough tokens to amortize:

```text
Eagle draft time + target verify time < accepted_tokens * normal_decode_time
```

With `eagle run2 wall ~= 10 ms` and `verify ~= 58 ms`, the break-even accepted
length is roughly:

```text
(10 + 58) / 46.7 ~= 1.46 steps
```

If comparing against 4 pure generated-token steps using `233.34 / 4 = 58.3 ms`,
the break-even is lower:

```text
(10 + 58) / 58.3 ~= 1.17 tokens
```

The exact conclusion depends on whether the prefill-after-logits step is counted
as part of the decode baseline. In all cases, accepted-token count is required
to evaluate Eagle accurately.

## Profile Logging Controls

Detailed per-op executor logs are disabled by default. Enable them only when
debugging dynamic quantization or kernel selection:

```bash
MNN_PROFILE_EXEC=1 \
LD_LIBRARY_PATH=/home/yydh/WAIC/mobiinfer/build_x86_opt \
./build_x86_opt/llm_demo <config.json> <prompt.txt>
```

This prints lines such as:

```text
[MNN_PROFILE_EXEC] op=/layers.0/mlp/down_proj/Linear CPU DenseConvInt8TiledExecutor ...
```

Eagle timing logs use a separate switch:

```bash
MNN_PROFILE_EAGLE_TIMING=1 \
LD_LIBRARY_PATH=/home/yydh/WAIC/mobiinfer/build_x86_opt \
./build_x86_opt/llm_demo <eagle_config.json> <prompt.txt>
```

For normal quality checks, leave both variables unset so generated text is not
mixed with profiling logs.

## Taobao Prompt Eagle Results

The reliable VLM quality/performance test should use the full prompt file,
because it already contains the agent instruction, the task, the output schema,
and the image tag:

```bash
LD_LIBRARY_PATH=/home/yydh/WAIC/mobiinfer/build_x86_opt \
./build_x86_opt/llm_demo \
  /mnt/e/WAIC/pc_server/models/Qwen3-VL-2B-Instruct-Eagle3-MNN/config.json \
  /mnt/e/WAIC/pc_server/test/data/example/prompts/taobao_mnn.txt
```

Do not use the `--image image question` demo path for Eagle quality decisions.
That path creates a much shorter prompt and does not match the phone-use agent
prompt format.

### 2026-06-24 Runs

Baseline non-Eagle:

```text
config: /mnt/e/WAIC/pc_server/models/Qwen3-VL-2B-Instruct-MNN/config.json
prompt tokens: 673
decode tokens: 108
vision time:   1.54 s
prefill time:  5.08 s
decode time:   4.38 s
decode speed: 24.64 tok/s
normal decode per token: 40.6 ms
```

Eagle depth=3:

```text
config: /mnt/e/WAIC/pc_server/models/Qwen3-VL-2B-Instruct-Eagle3-MNN/config.json
prompt tokens: 673
decode tokens: 135
vision time:   1.47 s
prefill time:  5.05 s
decode time:   8.09 s
decode speed: 16.68 tok/s
accept: 135 / 444 = 30.4%
verify steps: 111
avg accepted / step: 1.22
```

Eagle depth=1:

```text
config: logs/0624/eagle_depth1_model/config.json
prompt tokens: 673
decode tokens: 101
vision time:   1.77 s
prefill time:  6.74 s
decode time:   5.06 s
decode speed: 19.98 tok/s
accept: 102 / 168 = 60.7%
verify steps: 84
avg accepted / step: 1.21
```

Depth=1 has a higher accept ratio only because it drafts two tokens per step
instead of four. The useful metric is accepted tokens per verify step, and both
depths are about 1.2.

### Verify Cost Estimate

Using non-Eagle decode as the baseline:

```text
normal decode per token ~= 4.38 s / 108 = 40.6 ms
```

Approximate Eagle draft time is the sum of `[MNN_PROFILE_EAGLE]` timed stages.
The remaining decode time is target verify plus scheduling/readback overhead:

```text
depth=3:
  total decode:       8090.0 ms
  eagle draft timing:  822.6 ms, 7.41 ms/step
  verify+overhead:    7267.4 ms, 65.47 ms/step
  verify equivalent:  65.47 / 40.6 = 1.61 normal decode steps

depth=1:
  total decode:       5060.0 ms
  eagle draft timing:   12.3 ms, 0.15 ms/step
  verify+overhead:    5047.7 ms, 60.09 ms/step
  verify equivalent:  60.09 / 40.6 = 1.48 normal decode steps
```

Per output token:

```text
depth=3:
  eagle draft:       6.09 ms/token
  verify+overhead: 53.83 ms/token

depth=1:
  eagle draft:       0.12 ms/token
  verify+overhead: 49.98 ms/token
```

This explains why Eagle is slower on this prompt. A verify step costs about
1.5-1.6 normal decode tokens, but each verify accepts only about 1.2 tokens.
The break-even condition is not met. Depth=1 is less bad because the draft path
is almost free, but target verify still dominates.

## MAI Eagle Model Check

There is also a MAI Eagle MNN model under:

```text
/mnt/e/WAIC/pc_server/models/MAI-UI-2B-0422-instruct-1ep_RLv2_4NPUS_bs128_ds5050_step100-MNN-EAGLE-visual-nq-hqq-int4
```

It has both Eagle and no-spec configs:

```text
config.json
config_no_spec.json
```

The Eagle config uses depth=1:

```json
"speculative_type": "eagle",
"hidden_states": true,
"eagle_depth": 1,
"eagle_topk": 1
```

Commands:

```bash
MNN_PROFILE_EAGLE_TIMING=1 \
MNN_PROFILE_EXEC=1 \
LD_LIBRARY_PATH=/home/yydh/WAIC/mobiinfer/build_x86_opt \
./build_x86_opt/llm_demo \
  /mnt/e/WAIC/pc_server/models/MAI-UI-2B-0422-instruct-1ep_RLv2_4NPUS_bs128_ds5050_step100-MNN-EAGLE-visual-nq-hqq-int4/config.json \
  /mnt/e/WAIC/pc_server/test/data/example/prompts/taobao_mnn.txt \
  > logs/0624/local_taobao_mai_eagle_depth1_timing_exec_profile.log 2>&1

LD_LIBRARY_PATH=/home/yydh/WAIC/mobiinfer/build_x86_opt \
./build_x86_opt/llm_demo \
  /mnt/e/WAIC/pc_server/models/MAI-UI-2B-0422-instruct-1ep_RLv2_4NPUS_bs128_ds5050_step100-MNN-EAGLE-visual-nq-hqq-int4/config_no_spec.json \
  /mnt/e/WAIC/pc_server/test/data/example/prompts/taobao_mnn.txt \
  > logs/0624/local_taobao_mai_no_spec_baseline.log 2>&1
```

Results:

```text
MAI no-spec:
  prompt tokens: 673
  decode tokens: 512
  vision time:   1.66 s
  prefill time:  5.31 s
  decode time:  20.57 s
  decode speed: 24.89 tok/s
  normal decode per token: 40.18 ms

MAI Eagle depth=1:
  prompt tokens: 673
  decode tokens: 512
  vision time:   1.45 s
  prefill time:  5.09 s
  decode time:  13.14 s
  decode speed: 38.98 tok/s
  accept: 513 / 516 = 99.4%
  verify steps: 258
  avg accepted / step: 1.99
```

MAI Eagle depth=1 is faster on this prompt:

```text
decode time speedup: 20.57 / 13.14 = 1.57x
token/s speedup:     38.98 / 24.89 = 1.57x
```

Verify estimate:

```text
eagle timed stages: 38.64 ms total, 0.15 ms/step
verify+overhead: 13101.36 ms total, 50.78 ms/step
verify equivalent: 50.78 / 40.18 = 1.26 normal decode steps
```

Because depth=1 drafts two tokens per verify and accepts almost both tokens,
the average accepted length is 1.99, which is greater than the verify-equivalent
cost of 1.26 normal decode steps. That is why this MAI Eagle run is faster,
unlike the Qwen Eagle run above.

Quality caveat: both MAI runs reached 512 generated tokens and showed repetitive
text / malformed JSON patterns. This run is useful for speed and accept-rate
comparison, but should not be treated as a clean quality win.

## Current Optimization Priority

1. Investigate why `lm_head` uses 8-bit weights and `Int8GemmKernel` instead of W4.
2. Optimize decode M=1 W4 GEMM for MLP projections.
3. Reduce or optimize visual transformer GEMM cost if image latency is the target.
4. Re-run several times to confirm whether layer 16 is consistently slower.
5. Defer Raster optimization unless its share grows on ARM/Harmony.

## Useful Commands

Analyze current CSV:

```bash
python3 tools/profile_bottleneck.py logs/qwen3_vl_img_op_profile_phase.csv --top 40
```

Show all `lm_head` rows:

```bash
python3 - <<'PY'
import csv
for r in csv.DictReader(open('logs/qwen3_vl_img_op_profile_phase.csv')):
    if 'lm_head' in r['node_name']:
        print(r['node_name'], r['phase'], r['called_times'], r['gemm_m'], r['avg_ms'], r['weight_bits'], r['kernel'])
PY
```

Summarize decode projection cost:

```bash
python3 - <<'PY'
import csv, collections
acc=collections.defaultdict(float)
for r in csv.DictReader(open('logs/qwen3_vl_img_op_profile_phase.csv')):
    if r['phase'] != 'decode' or r['op_type'] != 'Convolution':
        continue
    name = r['node_name']
    if '/lm/lm_head' in name: key='lm_head'
    elif '/mlp/gate_proj' in name: key='mlp/gate_proj'
    elif '/mlp/up_proj' in name: key='mlp/up_proj'
    elif '/mlp/down_proj' in name: key='mlp/down_proj'
    elif '/self_attn/q_proj' in name: key='attn/q_proj'
    elif '/self_attn/k_proj' in name: key='attn/k_proj'
    elif '/self_attn/v_proj' in name: key='attn/v_proj'
    elif '/self_attn/o_proj' in name: key='attn/o_proj'
    else: key='other'
    acc[key] += float(r['avg_ms'] or 0)
for k, v in sorted(acc.items(), key=lambda x: x[1], reverse=True):
    print(f'{v:8.2f} ms {k}')
PY
```
