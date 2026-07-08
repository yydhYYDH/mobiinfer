# Token Count and Vocabulary Pruning Tools

This directory contains experimental tools for measuring tokenizer usage on prompt/message datasets and using the resulting token frequency table to prune a model vocabulary. The scripts were restored from `experiments/token_count_archive`.

The intended workflow is:

1. Count token usage on a representative dataset.
2. Prune the target/base model vocabulary with the generated `token_counts.csv`.
3. If using Eagle3 speculative decoding, prune the Eagle3 draft vocabulary from the target model's `old_to_new_token_id.json`.
4. Smoke test and benchmark the pruned model.

These scripts are model/tokenizer specific. Do not reuse token counts from a different tokenizer.

## Files

| File | Purpose |
|---|---|
| `count_prompt_tokens.py` | Count token frequencies for a parquet dataset with a `prompt` column. |
| `count_jsonl_message_tokens.py` | Count token frequencies for a JSONL chat dataset with `messages`. Single-process version. |
| `count_jsonl_token_counts_fast.py` | Multi-process JSONL token frequency counter. Usually the preferred counter for large JSONL datasets. |
| `count_jsonl_total_tokens_fast.py` | Multi-process JSONL total-token/length statistics only. Does not write a full token frequency table. |
| `prune_vocab_experiment.py` | Prune the target/base model tokenizer and embedding/lm head rows using `token_counts.csv`. |
| `prune_eagle3_vocab_from_target_mapping.py` | Prune Eagle3 draft vocab after target vocab pruning. Rebuilds `d2t`, `t2d`, and draft `lm_head.weight`. |
| `test_pruned_qwen3vl_taobao.py` | Run a Taobao image prompt through a pruned Qwen3VL-like model. |
| `benchmark_cpu_decode_pruned_vs_orig.py` | Compare original vs pruned model CPU generation timing. |
| `run_taobao_speed_repeats.py` | Repeat Taobao benchmark prompts and summarize speed. |

Extra notes are in `docs/`. Prompt samples are in `prompts/`.

## Environment

These scripts expect a Python environment with at least:

```bash
pip install torch transformers safetensors pandas pyarrow pillow tqdm
```

On the A100 server, the existing examples use the `mnn` conda environment:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate mnn
```

## Count JSONL Message Tokens

For large JSONL chat datasets, use the fast counter:

```bash
python tools/token_count/count_jsonl_token_counts_fast.py \
  --data /path/to/train.jsonl \
  --model /path/to/hf_model_or_tokenizer \
  --out-dir /path/to/token_count_outputs \
  --workers 16 \
  --batch-size 512 \
  --top-k 500
```

Expected input format is one JSON object per line with a `messages` field. Each message should contain `role` and `content`.

Outputs include:

- `token_counts.csv`: sorted token frequency table with `token_id,token,count`.
- `unused_tokens.csv`: tokenizer ids not observed in the dataset.
- `top_<K>_tokens.txt`: human-readable top token list.
- `summary.json`: row count, total tokens, length percentiles, role counts, special token counts.

Use `--no-chat-template` only when you intentionally want to tokenize concatenated role/content text instead of the tokenizer chat template.

For quick length-only statistics without a full token frequency table:

```bash
python tools/token_count/count_jsonl_total_tokens_fast.py \
  --data /path/to/train.jsonl \
  --model /path/to/hf_model_or_tokenizer \
  --out-dir /path/to/jsonl_total_outputs \
  --workers 16 \
  --batch-size 512
```

## Count Parquet Prompt Tokens

For a parquet dataset with a `prompt` column:

```bash
python tools/token_count/count_prompt_tokens.py \
  --data /path/to/train.parquet \
  --model /path/to/hf_model_or_tokenizer \
  --out-dir /path/to/prompt_token_outputs \
  --batch-size 1024 \
  --top-k 500
```

This writes the same main artifacts: `token_counts.csv`, `top_<K>_tokens.txt`, and `summary.json`.

## Prune Target/Base Vocabulary

Run a dry run first:

```bash
python tools/token_count/prune_vocab_experiment.py \
  --model /path/to/original_hf_model \
  --counts /path/to/token_counts.csv \
  --out /path/to/model_vocab_pruned \
  --dry-run
```

Then run the actual prune:

```bash
python tools/token_count/prune_vocab_experiment.py \
  --model /path/to/original_hf_model \
  --counts /path/to/token_counts.csv \
  --out /path/to/model_vocab_pruned
```

The script copies model metadata and rewrites:

- `tokenizer.json`
- `added_tokens.json`
- `tokenizer_config.json`
- `config.json`
- `generation_config.json`, if present
- `model.safetensors`

It also writes:

- `old_to_new_token_id.json`
- `kept_old_token_ids.txt`

The target script currently force-keeps a few token ids used by a Taobao prompt formatting case. Those ids are tokenizer-specific. Before using this script on a new tokenizer family, verify or remove that force-keep list in `build_keep()`.

## Existing Token Count Artifacts

The archive currently contains several reusable `token_counts.csv` files under `experiments/token_count_archive/`.

Preferred full JSONL count for the MAI/UI-Venus tokenizer family:

```text
experiments/token_count_archive/jsonl_token_count_outputs/token_counts.csv
```

Summary:

- Data: `/temp/csm/Dataset-train/mobimind_e2e_train.jsonl`
- Tokenizer/model used for counting: `/temp/zhangdelong/models/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40`
- Rows processed: `2,258,630`
- Total message tokens: `2,464,319,155`
- Vocab size reported by the counter: `151,669`
- Used token ids: `14,162`
- Chat template: enabled

Full parquet prompt count:

```text
experiments/token_count_archive/outputs/token_counts.csv
```

Summary:

- Data: `/temp/zhangdelong/data/data-0422-instruct-halfpixel/train.parquet`
- Tokenizer/model used for counting: `/temp/zhangdelong/models/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40`
- Rows processed: `48,976`
- Total prompt tokens: `35,037,086`
- Used token ids: `9,236`
- Chat template: enabled

Sample-only counts also exist, but should not be used for production pruning:

```text
experiments/token_count_archive/jsonl_sample_out/token_counts.csv
experiments/token_count_archive/jsonl_token_count_sample_out/token_counts.csv
experiments/token_count_archive/sample_out/token_counts.csv
```

Before reusing a count file for a different model checkpoint, verify tokenizer identity. For example, the following target model has the same tokenizer files as the counting model:

```text
/home/ma-user/modelarts/user-job-dir/lzz/models/UI-Venus-1.5-2B-0422-reasoning-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step100
```

Verified identical files:

- `tokenizer.json`
- `vocab.json`
- `merges.txt`
- `added_tokens.json`
- `special_tokens_map.json`
- `tokenizer_config.json`
- `chat_template.jinja`

Dry-run command for this model:

```bash
MODEL_PATH=/home/ma-user/modelarts/user-job-dir/lzz/models/UI-Venus-1.5-2B-0422-reasoning-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step100
COUNTS=experiments/token_count_archive/jsonl_token_count_outputs/token_counts.csv
OUT=${MODEL_PATH}_vocab_pruned

python tools/token_count/prune_vocab_experiment.py \
  --model "$MODEL_PATH" \
  --counts "$COUNTS" \
  --out "$OUT" \
  --dry-run
```

Actual prune command:

```bash
MODEL_PATH=/home/ma-user/modelarts/user-job-dir/lzz/models/UI-Venus-1.5-2B-0422-reasoning-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step100
COUNTS=experiments/token_count_archive/jsonl_token_count_outputs/token_counts.csv
OUT=${MODEL_PATH}_vocab_pruned

python tools/token_count/prune_vocab_experiment.py \
  --model "$MODEL_PATH" \
  --counts "$COUNTS" \
  --out "$OUT"
```

## Prune Eagle3 Vocabulary

After target vocab pruning, Eagle3 mappings must be rebuilt. Target token ids have changed, so reusing the original Eagle3 `d2t/t2d` is incorrect.

Dry run:

```bash
python tools/token_count/prune_eagle3_vocab_from_target_mapping.py \
  --eagle-model /path/to/original_eagle3_model \
  --target-old-to-new /path/to/model_vocab_pruned/old_to_new_token_id.json \
  --target-config /path/to/model_vocab_pruned/config.json \
  --out-eagle /path/to/eagle3_vocab_pruned \
  --dry-run
```

Actual run:

```bash
python tools/token_count/prune_eagle3_vocab_from_target_mapping.py \
  --eagle-model /path/to/original_eagle3_model \
  --target-old-to-new /path/to/model_vocab_pruned/old_to_new_token_id.json \
  --target-config /path/to/model_vocab_pruned/config.json \
  --out-eagle /path/to/eagle3_vocab_pruned
```

Outputs include:

- `model.safetensors`
- `config.json`
- `eagle_token_mapping.csv`
- `kept_old_draft_token_ids.txt`

Optional further filtering by target token frequency:

```bash
python tools/token_count/prune_eagle3_vocab_from_target_mapping.py \
  --eagle-model /path/to/original_eagle3_model \
  --target-old-to-new /path/to/model_vocab_pruned/old_to_new_token_id.json \
  --target-config /path/to/model_vocab_pruned/config.json \
  --target-counts /path/to/token_counts.csv \
  --min-count 10 \
  --out-eagle /path/to/eagle3_vocab_pruned_count10
```

## Smoke Test and Benchmark

For Qwen3VL/Taobao-style local tests:

```bash
python tools/token_count/test_pruned_qwen3vl_taobao.py \
  --model /path/to/model_vocab_pruned \
  --md tools/token_count/prompts/taobao_short_text_img.md \
  --image /path/to/image.jpg \
  --max-new-tokens 64 \
  --device cpu
```

Compare original vs pruned decode timing:

```bash
python tools/token_count/benchmark_cpu_decode_pruned_vs_orig.py \
  --md tools/token_count/prompts/taobao_short_text_img.md \
  --image /path/to/image.jpg \
  --max-new-tokens 64 \
  --dtype bf16 \
  --order orig-first
```

For MNN export and C++ runtime smoke tests with Eagle3, see `docs/prune_qwen3vl_eagle_vocab.md`.

## Important Constraints

- Token counts must be generated by the same tokenizer as the model being pruned.
- Always keep special tokens and config token ids.
- BPE pruning must preserve merge dependencies for kept tokens.
- Target vocab pruning changes token ids; every downstream mapping that stores token ids must be remapped.
- For Eagle3, `target_token_id = draft_token_id + d2t[draft_token_id]`; after target pruning, `lm_head.weight`, `d2t`, `t2d`, and `draft_vocab_size` must be rebuilt together.
- Successful export is not enough. Run a runtime smoke test before trusting benchmark results.
