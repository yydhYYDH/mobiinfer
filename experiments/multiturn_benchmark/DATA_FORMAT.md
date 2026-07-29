# Data Format

The source dataset is `/temp/csm/Dataset-train/mobimind_e2e_train.jsonl`.
`prepare_multiturn_prompts.py` groups consecutive GUI-agent decisions into
trajectories without modifying the source data.

Generated benchmark data contains `manifest.tsv`, `summary.json`, flattened
`prompts/trajectory_XXX/step_YYY.txt`, and optionally copied screenshots under
`images/`. The manifest records trajectory ID, step ID, history length, prompt
path, image path, task text, and the gold assistant JSON.

Experiment output contains one log per matrix cell, row-level `results.csv`, and
aggregated `summary.json`. Memory sampling additionally writes a JSON summary
and a time-series CSV containing RSS, PSS, USS, and SwapPSS values in bytes.

Generated prompts, screenshots, binaries, Python caches, and raw logs are not
tracked. Compact CSV/JSON summaries supporting reported results may be tracked.
