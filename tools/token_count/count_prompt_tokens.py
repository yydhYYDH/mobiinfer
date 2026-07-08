#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import Counter

import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_DATA = "/temp/zhangdelong/data/data-0422-instruct-halfpixel/train.parquet"
DEFAULT_MODEL = "/temp/zhangdelong/models/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40"


def normalize_prompt(prompt):
    return [{"role": item["role"], "content": item["content"]} for item in prompt]


def encode_prompt(tokenizer, prompt, use_chat_template=True):
    messages = normalize_prompt(prompt)
    if use_chat_template:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        except Exception:
            # Fall back for tokenizer configs without a usable chat template.
            pass
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return tokenizer.encode(text, add_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Count prompt token frequencies for a parquet dataset.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to train.parquet")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model/tokenizer directory")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=256, help="Parquet prompt batch size")
    parser.add_argument("--top-k", type=int, default=200, help="Number of frequent tokens to print/write")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows")
    parser.add_argument("--no-chat-template", action="store_true", help="Tokenize concatenated message text directly")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    parquet = pq.ParquetFile(args.data)
    total_rows = parquet.metadata.num_rows
    rows_to_process = min(total_rows, args.limit) if args.limit else total_rows

    counter = Counter()
    per_row_lengths = []
    processed = 0

    with tqdm(total=rows_to_process, desc="tokenizing prompts", unit="row") as progress:
        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=["prompt"]):
            prompts = batch.column(0).to_pylist()
            for prompt in prompts:
                if args.limit is not None and processed >= args.limit:
                    break
                token_ids = encode_prompt(tokenizer, prompt, use_chat_template=not args.no_chat_template)
                counter.update(token_ids)
                per_row_lengths.append(len(token_ids))
                processed += 1
                progress.update(1)
            if args.limit is not None and processed >= args.limit:
                break

    token_counts_path = os.path.join(args.out_dir, "token_counts.csv")
    with open(token_counts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "token", "count"])
        for token_id, count in counter.most_common():
            writer.writerow([token_id, tokenizer.convert_ids_to_tokens(token_id), count])

    top_tokens_path = os.path.join(args.out_dir, f"top_{args.top_k}_tokens.txt")
    with open(top_tokens_path, "w", encoding="utf-8") as f:
        for rank, (token_id, count) in enumerate(counter.most_common(args.top_k), start=1):
            token = tokenizer.convert_ids_to_tokens(token_id)
            f.write(f"{rank:5d}  id={token_id:<8d} count={count:<10d} token={token!r}\n")

    total_tokens = sum(counter.values())
    vocab_size = len(tokenizer)
    used_token_count = len(counter)
    unused_token_count = vocab_size - used_token_count
    sorted_lengths = sorted(per_row_lengths)

    def percentile(p):
        if not sorted_lengths:
            return 0
        idx = min(len(sorted_lengths) - 1, int(round((len(sorted_lengths) - 1) * p / 100.0)))
        return sorted_lengths[idx]

    special_token_ids = set(tokenizer.all_special_ids)
    special_token_counts = {
        str(token_id): {
            "token": tokenizer.convert_ids_to_tokens(token_id),
            "count": counter.get(token_id, 0),
        }
        for token_id in sorted(special_token_ids)
    }

    summary = {
        "data": args.data,
        "model": args.model,
        "rows_in_parquet": total_rows,
        "rows_processed": processed,
        "total_prompt_tokens": total_tokens,
        "vocab_size": vocab_size,
        "used_token_count": used_token_count,
        "unused_token_count": unused_token_count,
        "avg_tokens_per_prompt": (total_tokens / processed) if processed else 0,
        "min_tokens_per_prompt": sorted_lengths[0] if sorted_lengths else 0,
        "max_tokens_per_prompt": sorted_lengths[-1] if sorted_lengths else 0,
        "p50_tokens_per_prompt": percentile(50),
        "p90_tokens_per_prompt": percentile(90),
        "p95_tokens_per_prompt": percentile(95),
        "p99_tokens_per_prompt": percentile(99),
        "special_token_counts": special_token_counts,
        "used_chat_template": not args.no_chat_template,
        "token_counts_csv": token_counts_path,
        "top_tokens_txt": top_tokens_path,
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTop tokens:")
    for rank, (token_id, count) in enumerate(counter.most_common(min(args.top_k, 30)), start=1):
        token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"{rank:5d}  id={token_id:<8d} count={count:<10d} token={token!r}")


if __name__ == "__main__":
    main()
