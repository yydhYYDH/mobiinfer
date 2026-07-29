# Phone Multi-turn Ablation Pilot

## Scope

- Device: OnePlus 13
- Source commit: `426e9a43`
- Tasks: trajectories `012` and `019`
- Controlled trajectory prefixes: 1, 2, 3, 4, and 5 steps
- Maximum new tokens: 192
- Pilot repeats: 1 per matrix cell
- Run order: deterministic shuffle per step length
- Cooldown: 10 seconds between cells

The same two 5-step trajectories are truncated to each requested length. This
keeps task composition fixed when measuring how the benefit changes with the
number of interaction steps.

## Ablations

| Variant | Vocabulary | Decoding | History KV |
| --- | --- | --- | --- |
| `baseline` | Full | AR | No |
| `kv_only` | Full | AR | Yes |
| `prune_only` | Pruned | AR | No |
| `prune_kv` | Pruned | AR | Yes |
| `prune_d3` | Pruned | D3 | No |
| `full` | Pruned | D3 | Yes |

## Quality Checks

All 30 pilot cells completed with:

- `errors = 0`
- clean JSON action rate = 100%
- gold action match rate = 100%
- token-boundary fallback count = 0

## Step-length Curve

Times aggregate the two fixed trajectories. End-to-end time is tokenizer and
vision time plus generation wall time.

| Steps | Baseline E2E (s) | Full E2E (s) | E2E speedup | Prefill speedup | Decode throughput speedup | Prefill token reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 22.93 | 13.65 | 1.68x | 1.37x | 2.69x | 0.0% |
| 2 | 48.90 | 27.47 | 1.78x | 1.73x | 2.42x | 32.7% |
| 3 | 102.76 | 52.08 | 1.97x | 1.99x | 2.73x | 45.0% |
| 4 | 143.78 | 68.54 | 2.10x | 2.25x | 2.69x | 52.5% |
| 5 | 193.05 | 84.77 | 2.28x | 2.73x | 2.79x | 57.9% |

The increasing E2E speedup is the expected signature of growing-history KV
reuse: longer trajectories amortize more repeated prompt prefill.

## Five-step Pilot

| Variant | E2E (s) | Prefill (s) | Decode (s) | Actual prefill tokens | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 193.05 | 94.50 | 72.14 | 8932 | 15.55 |
| `kv_only` | 126.28 | 37.65 | 65.10 | 3766 | 17.56 |
| `prune_only` | 186.00 | 92.94 | 67.65 | 8942 | 16.98 |
| `prune_kv` | 212.39 | 41.91 | 144.99 | 3768 | 8.06 |
| `prune_d3` | 160.77 | 89.85 | 45.23 | 8942 | 25.14 |
| `full` | 84.77 | 34.65 | 26.36 | 3768 | 43.32 |

The first `prune_kv` measurement was a thermal/runtime outlier. Three targeted
repeats produced E2E times of `96.87`, `112.17`, and `116.68` seconds, with a
median of `112.17` seconds. Each repeat retained 100% clean JSON and gold action
match. The single-run value `212.39` must not be used in a paper figure.

## Paper Run

The pilot is useful for validating the experiment design, but it is not the
final paper result. Run at least three repeats and report medians and dispersion:

```bash
cd /data/data/com.termux/files/home/csm/mobiinfer
python experiments/multiturn_benchmark/run_phone_ablation_matrix.py \
  --output bench_multiturn_ablation_paper_r3 \
  --binary experiments/multiturn_benchmark/bin_ablation_v2/multiturn_bench_demo \
  --tasks 2 \
  --repeats 3 \
  --cooldown-seconds 20
```

The current benchmark contains only two 5-step trajectories. Use this as a
controlled latency microbenchmark. For a stronger task-level result, extract
more long trajectories from the source dataset before making a general claim
about 5-step accuracy.

## Artifacts

- Runner: `experiments/multiturn_benchmark/run_phone_ablation_matrix.py`
- Pilot rows: `results/phone_ablation_426e9a43_run1/results.csv`
- Pilot summary: `results/phone_ablation_426e9a43_run1/summary.json`
- Outlier repeats: `results/phone_ablation_v2_prune_kv_s5_r3/results.csv`
