# Ngram Lookup Table

This directory contains scripts for building assistant-output ngram lookup tables for MNN lookahead decoding.

## Dataset

```text
/temp/csm/Dataset-train/mobimind_e2e_train.jsonl
```

Only `assistant` messages are counted.

## Tokenizer / Model

```text
/temp/zhangdelong/models/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40
```

The scripts use the model tokenizer, so token ids match the runtime model.

## Scripts

```text
build_assistant_ngram_lookup.py   # build full 1..8 gram frequency tables
filter_ngram_lookup.py            # filter min_count=20 and keep top1/top3 next token per key
```

## Outputs

Large generated files are stored under:

```text
artifacts/lookup_tables/ngram/
```

Important filtered outputs:

```text
artifacts/lookup_tables/ngram/filtered/assistant_ngram_min20_top1.tsv.gz
artifacts/lookup_tables/ngram/filtered/assistant_ngram_min20_top3.tsv.gz
artifacts/lookup_tables/ngram/filtered/assistant_ngram_min20_top_summary.json
```

For MNN lookahead speedup, start with:

```text
assistant_ngram_min20_top1.tsv.gz
```

Rationale: MNN currently selects one draft token for each matched ngram step. `top1` keeps the table smaller and avoids low-confidence successors that are likely to fail verification.

## TSV Format

```text
n	key_token_ids	next_token_id	count
```

Example:

```text
8	12,34,56,78,90,11,22,33	44	128
```

This means the 8-token key is followed by token `44` 128 times in assistant outputs.

## Remap Token IDs For Vocab-Pruned Models

Use `remap_ngram_token_ids.py` when a model keeps the same tokenizer semantics but changes token ids through a vocab-pruning map such as `old_to_new_token_id.json`.

```bash
python tools/ngram/remap_ngram_token_ids.py \
  --input artifacts/lookahead_bench/ngram_trials/assistant_min20_top1_n1_4.tsv \
  --mapping /temp/csm/autoround_export/UI-Venus-1.5-2B-0422-reasoning-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step100_vocab_pruned/old_to_new_token_id.json \
  --output artifacts/lookahead_bench/ngram_trials/assistant_min20_top1_n1_4_vocab_pruned.tsv \
  --max-key-len 4 \
  --min-count 20 \
  --top-k 1
```

Rows containing a token id that is not present in the mapping are dropped. Duplicate remapped `(n, key, next_token)` rows are merged by summing counts, then candidates are filtered and ranked per key.

## Experimental Mmap-Direct Hash Binary

`build_mmap_hash_ngram.py` builds an experimental V3 binary format for top1 ngram tables. It stores separate sections for each n and a static hash-bucket index, so a runtime can mmap the file and look up keys directly without first building an `unordered_map`.

```bash
python tools/ngram/build_mmap_hash_ngram.py \
  --input artifacts/lookahead_bench/ngram_trials/assistant_min20_top1_n1_4_vocab_pruned.tsv \
  --output artifacts/lookahead_bench/ngram_trials/assistant_min20_top1_n1_4_vocab_pruned.hash.mnnngram3 \
  --max-key-len 4 \
  --load-factor 2.0
```

Benchmark the Python mmap reader:

```bash
python tools/ngram/bench_mmap_hash_ngram.py \
  --binary artifacts/lookahead_bench/ngram_trials/assistant_min20_top1_n1_4_vocab_pruned.hash.mnnngram3 \
  --queries artifacts/lookahead_bench/ngram_trials/assistant_min20_top1_n1_4_vocab_pruned.tsv \
  --max-key-len 4 \
  --lookups 1000000
```

This V3 prototype is not wired into MNN runtime yet. It is for validating layout size and lookup behavior before adding the C++ loader.
