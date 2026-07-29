# Results

The validated phone pilot used source commit `426e9a43`, two fixed trajectories,
step lengths 1 through 5, and six ablation variants. All 30 cells completed with
zero errors, 100% clean JSON actions, 100% gold action match, and no
token-boundary fallback.

For the five-step comparison, the baseline took 193.05 s end to end and the full
configuration took 84.77 s, a 2.28x speedup. Actual prefill tokens fell by 57.9%,
and decode throughput increased from 15.55 to 43.32 token/s. One `prune_kv`
pilot cell was a thermal/runtime outlier; its targeted three-repeat median was
112.17 s and must be used instead of the original 212.39 s cell.

Across three memory repeats, the full configuration reduced median peak PSS by
29.77% (1185.75 MiB) on `sai-a100` and 31.82% (1199.72 MiB) on OnePlus 13.
No run used swap and all measured inference logs completed without errors.

Detailed latency tables and caveats are in `PHONE_ABLATION_PILOT.md`. Memory
tables are in `PEAK_MEMORY_RESULTS.md`.
