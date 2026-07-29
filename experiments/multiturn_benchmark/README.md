# Multi-turn MobiMind Benchmark

This directory prepares and runs a replay-style multi-turn benchmark for
mobile GUI-agent inference.

## Why this benchmark exists

The existing `bench20` phone scripts run independent decision prompts:

```text
llm_demo config.json prompt_XX.txt MAX_NEW
```

Each `prompt_XX.txt` may contain a long `Action History`, but the benchmark does
not execute a single task trajectory step by step. This hides the workload shape
we care about: across one task, the system prompt, action space, task goal, and
most action history are shared with later steps, while the current screenshot and
the newest history entry change.

This benchmark extracts consecutive decision steps from the MobiMind JSONL file
and writes them as trajectories:

```text
trajectory_000/
  step_000.txt
  step_001.txt
  ...
```

Each step prompt is still a full prompt, so it can be run by the current
`llm_demo` without changing C++ code. The generated layout also makes it easy to
add a session-based runner later that keeps the KV cache for the fixed prefix
and the growing action history.

## Input

Default dataset:

```text
/temp/csm/Dataset-train/mobimind_e2e_train.jsonl
```

Each JSONL row should contain:

- `messages`: usually `system`, `user`, and `assistant`.
- `images`: image path list for the current screen observation.

The script also supports ShareGPT-style JSON arrays such as:

```text
/temp/csm/sft-0422-reasoning-quant-20k-half-size/mobimind_calibration_20k.json
```

For shuffled calibration files, use `--group-by-task` to rebuild trajectories by
grouping samples with the same `### Current Task` and sorting consecutive
`Action History` lengths.

Trajectory continuity is checked strictly: for step `k`, the prompt must contain
exactly `k` history entries, and the last history entry must match the assistant
JSON from step `k-1`. This avoids grouping unrelated samples that share the same
task text.

The `user` message is expected to contain:

```text
### Current Task
"..."
### Action History
...
<image>
...
Please provide the next action ...
```

## Generate prompts on the remote server

```bash
cd /home/ma-user/workspace/csm/mobiinfer
python3 experiments/multiturn_benchmark/prepare_multiturn_prompts.py \
  --jsonl /temp/csm/Dataset-train/mobimind_e2e_train.jsonl \
  --output /tmp/mobimind_multiturn_bench \
  --phone-root /data/data/com.termux/files/home/csm/mobiinfer/multiturn_bench \
  --num-trajectories 20 \
  --min-steps 3 \
  --max-steps 12 \
  --copy-images
```

For the 20k calibration file:

```bash
cd /home/ma-user/workspace/csm/mobiinfer
python3 experiments/multiturn_benchmark/prepare_multiturn_prompts.py \
  --jsonl /temp/csm/sft-0422-reasoning-quant-20k-half-size/mobimind_calibration_20k.json \
  --output /tmp/mobimind_multiturn_calib \
  --phone-root /data/data/com.termux/files/home/csm/mobiinfer/multiturn_bench \
  --num-trajectories 20 \
  --min-steps 3 \
  --max-steps 12 \
  --copy-images \
  --image-source-dir /temp/csm/sft-0422-reasoning-quant-20k-half-size \
  --group-by-task
```

Note: the 20k calibration file is useful for single-step calibration and prompt
profiling, but it may not contain adjacent decision states from the same
episode. If strict continuity is required, prefer the original training JSONL:

```bash
cd /home/ma-user/workspace/csm/mobiinfer
python3 experiments/multiturn_benchmark/prepare_multiturn_prompts.py \
  --jsonl /temp/csm/Dataset-train/mobimind_e2e_train.jsonl \
  --output /tmp/mobimind_multiturn_train_strict \
  --phone-root /data/data/com.termux/files/home/csm/mobiinfer/multiturn_bench \
  --num-trajectories 20 \
  --min-steps 3 \
  --max-steps 5 \
  --group-by-task
```

The output contains:

- `manifest.tsv`: one row per decision step.
- `summary.json`: task/step/image statistics.
- `prompts/trajectory_XXX/step_YYY.txt`: flattened prompts for `llm_demo`.
- `images/`: selected screenshots, if `--copy-images` is set.

## Copy to phone

```bash
ssh oneplus13 'mkdir -p /data/data/com.termux/files/home/csm/mobiinfer/multiturn_bench'
scp -r /tmp/mobimind_multiturn_bench/* \
  oneplus13:/data/data/com.termux/files/home/csm/mobiinfer/multiturn_bench/
```

## Run on phone

Copy `run_phone_multiturn_raw.sh` to the phone under:

```text
/data/data/com.termux/files/home/csm/mobiinfer/multiturn_bench/
```

Then run:

```bash
cd /data/data/com.termux/files/home/csm/mobiinfer/multiturn_bench
ROOT=/data/data/com.termux/files/home/csm/mobiinfer \
BENCH=/data/data/com.termux/files/home/csm/mobiinfer/multiturn_bench \
MAX_NEW=192 \
./run_phone_multiturn_raw.sh
```

This raw runner launches a fresh `llm_demo` process for every step. It is the
correct baseline for measuring the current benchmark behavior. It does not yet
reuse KV cache across steps.

## Cache interpretation

The `Action History` grows over the trajectory. In a cache-aware runner, the
recommended prompt split is:

```text
cached base:
  System Prompt
  Current Task
  Action History header

growing cached history:
  1. previous assistant JSON
  2. previous assistant JSON
  ...

per-step dynamic suffix:
  current screenshot
  next-action instruction
```

After each step, the runner erases only the dynamic suffix and generated tokens,
appends the gold action to the cached history, and then appends the next
screenshot/instruction. `multiturn_bench_demo` implements this session-based
path and reports both logical prompt tokens and tokens actually prefetched.

## Build the continuous-session runner

Activate the server environment and build the MNN LLM demo targets:

```bash
cd /home/ma-user/workspace/csm/mobiinfer
conda activate mnn
mkdir -p build && cd build
cmake .. -DMNN_BUILD_LLM=ON
cmake --build . --target multiturn_bench_demo -j$(nproc)
```

The runner accepts four modes:

- `raw-ar`: full-prompt autoregressive baseline with no history KV reuse.
- `raw`: full-prompt run using the decoding method in the selected config.
- `cached-history`: reuse the fixed prefix and growing action-history KV.
- `cached-base`: reuse only the fixed base prefix; useful for diagnosis.

Run a single variant as follows:

```bash
./build/bin/multiturn_bench_demo \
  bench20/configs/config_vocab_pruned_w8a8_la_d3_bin.json \
  bench_multiturn_20traj_4step \
  cached-history \
  192
```

On the phone, use the same argument order after copying the binary, configs,
models, n-gram table, and generated benchmark directory under `$ROOT`.

## Run the phone ablation matrix

`run_phone_ablation_matrix.py` compares the contributions from vocabulary
pruning, speculative decoding, and history KV reuse while keeping the selected
trajectory prefixes fixed:

```bash
cd /data/data/com.termux/files/home/csm/mobiinfer
python experiments/multiturn_benchmark/run_phone_ablation_matrix.py \
  --output bench_multiturn_ablation_paper_r3 \
  --binary experiments/multiturn_benchmark/bin_ablation_v2/multiturn_bench_demo \
  --tasks 2 \
  --repeats 3 \
  --cooldown-seconds 20
```

The six variants are `baseline`, `kv_only`, `prune_only`, `prune_kv`,
`prune_d3`, and `full`. Each run writes individual logs, `results.csv`, and
`summary.json`; existing output directories are not overwritten.

## Validate output quality

Check that generated actions remain parseable and match the recorded gold
actions before using timing results:

```bash
python experiments/multiturn_benchmark/verify_model_outputs.py \
  bench_multiturn_ablation_paper_r3/logs/r00_s5_full.log \
  --expected-steps 10 \
  --manifest bench_multiturn_20traj_4step/manifest.tsv
```

For the validated pilot, all 30 matrix cells completed without errors, produced
clean JSON, and matched the gold action. On two fixed five-step trajectories,
the full configuration reduced end-to-end time from 193.05 s to 84.77 s
(2.28x), reduced actual prefill tokens by 57.9%, and increased decode throughput
from 15.55 to 43.32 token/s. Treat these as controlled microbenchmark results,
not a dataset-wide accuracy claim. See `PHONE_ABLATION_PILOT.md` for the full
step-length curve and outlier notes.

## Measure peak memory

Measure the complete process lifetime, including model loading, vision, prefill,
and decode:

```bash
python experiments/multiturn_benchmark/measure_peak_memory.py \
  --help
```

The recorded three-repeat comparison found a median peak PSS reduction of
29.77% on `sai-a100` and 31.82% on OnePlus 13 for the full configuration versus
the baseline. See `PEAK_MEMORY_RESULTS.md` and
`results/peak_memory_combined_20260723/summary.csv`.

## Tracked benchmark artifacts

- `prepare_multiturn_prompts.py`: extract continuous trajectories.
- `run_phone_multiturn_raw.sh`: fresh-process baseline runner.
- `run_phone_ablation_matrix.py`: controlled phone ablation runner.
- `summarize_multiturn_logs.py`: aggregate raw replay logs.
- `verify_model_outputs.py`: format and gold-action checks.
- `measure_peak_memory.py`: RSS/PSS/USS sampling.
- `PHONE_ABLATION_PILOT.md`: latency pilot and caveats.
- `PEAK_MEMORY_RESULTS.md`: server and phone memory results.

Generated benchmark inputs, copied screenshots, build products, Python cache
directories, and large raw log trees should remain outside Git. Track compact
CSV/JSON summaries needed to support reported numbers.
