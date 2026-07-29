# Reproduce

Run commands from `/home/ma-user/workspace/csm/mobiinfer`. Activate the server
environment with `conda activate mnn` before building or server-side tests.

Generate benchmark inputs with the `prepare_multiturn_prompts.py` command in
`README.md`, copy the generated directory and required runtime assets to the
phone, then run the controlled matrix:

```bash
python experiments/multiturn_benchmark/run_phone_ablation_matrix.py \
  --output bench_multiturn_ablation_paper_r3 \
  --binary experiments/multiturn_benchmark/bin_ablation_v2/multiturn_bench_demo \
  --tasks 2 \
  --repeats 3 \
  --cooldown-seconds 20
```

Validate each result log using `verify_model_outputs.py`, passing the expected
number of output blocks and the source `manifest.tsv`. Use
`measure_peak_memory.py --help` for the process-lifetime memory wrapper. Full
build, raw-runner, validation, and artifact descriptions are in `README.md`.
