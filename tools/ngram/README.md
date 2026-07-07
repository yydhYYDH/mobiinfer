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
