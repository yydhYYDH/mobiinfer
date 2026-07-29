# Peak Memory Results

## Setup

- Workload: the same continuous 4-step trajectory (`000`)
- Baseline: full vocabulary, W8A8 AR, full-prompt prefill
- Full: pruned vocabulary and LM head, D3, and history KV reuse
- Repeats: 3 per platform and variant
- Sampling interval: 200 ms
- Scope: process launch, model loading, vision, prefill, and decode
- Platforms: OnePlus 13 and `sai-a100`

The monitor reads `VmHWM` from `/proc/<pid>/status` and samples
`/proc/<pid>/smaps_rollup`. Peak PSS is the primary paper metric. Peak RSS is
reported as a more stable process-level cross-check. Values are MiB.

## Results

| Platform | Variant | Peak RSS median [min, max] | Peak PSS median [min, max] | Peak USS median [min, max] |
| --- | --- | ---: | ---: | ---: |
| Server | Baseline | 3986.82 [3986.79, 3986.86] | 3983.15 [3983.12, 3983.19] | 3983.12 [3983.08, 3983.15] |
| Server | Full | 2800.80 [2799.91, 2801.47] | 2797.40 [2796.97, 2797.70] | 2797.37 [2796.93, 2797.66] |
| Phone | Baseline | 3879.19 [3878.70, 3879.38] | 3769.98 [3691.82, 3859.23] | 3767.05 [3688.90, 3856.31] |
| Phone | Full | 2626.07 [2625.78, 2626.43] | 2570.26 [2570.24, 2578.77] | 2567.36 [2567.30, 2575.80] |

## Reduction

| Platform | Peak RSS reduction | Peak PSS reduction | Peak PSS saved |
| --- | ---: | ---: | ---: |
| Server | 29.75% | 29.77% | 1185.75 MiB |
| Phone | 32.30% | 31.82% | 1199.72 MiB |

No run used swap (`Peak SwapPSS = 0`). All 12 inference logs completed four
steps with `errors = 0` and `token boundary fallbacks = 0`.

Android PSS varies more than RSS because proportional ownership of shared and
file-backed pages depends on other live processes. The phone RSS results are
stable across repeats and support the same conclusion as PSS.

## Artifacts

- Monitor: `measure_peak_memory.py`
- Tracked combined summary: `results/peak_memory_combined_20260723/summary.csv`
- Server raw JSON/log/sample series: `results/peak_memory_server_20260723/`
- Phone raw JSON/log/sample series: retained with the corresponding phone run
  and intentionally excluded from Git.
