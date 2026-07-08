#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import Counter

from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_DATA = "/temp/csm/Dataset-train/mobimind_e2e_train.jsonl"
DEFAULT_MODEL = "/temp/zhangdelong/models/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40"


def encode_messages(tokenizer, messages, use_chat_template=True):
    normalized = [{"role": item["role"], "content": item["content"]} for item in messages]
    if use_chat_template:
        try:
            return tokenizer.apply_chat_template(normalized, tokenize=True, add_generation_prompt=False)
        except Exception:
            pass
    text = "\n".join(f"{m['role']}: {m['content']}" for m in normalized)
    return tokenizer.encode(text, add_special_tokens=True)


def count_lines(path):
    total = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def main():
    parser = argparse.ArgumentParser(description="Count chat-template token frequencies for a JSONL messages dataset.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default="jsonl_outputs")
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--skip-line-count", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    rows_total = args.limit if args.limit is not None else (None if args.skip_line_count else count_lines(args.data))

    token_counter = Counter()
    role_counter = Counter()
    length_counter = []
    processed = 0

    with open(args.data, encoding="utf-8") as f:
        progress = tqdm(total=rows_total, desc="tokenizing jsonl messages", unit="row")
        with progress:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                messages = obj["messages"]
                for message in messages:
                    role_counter[message.get("role", "")] += 1
                token_ids = encode_messages(tokenizer, messages, use_chat_template=not args.no_chat_template)
                token_counter.update(token_ids)
                length_counter.append(len(token_ids))
                processed += 1
                progress.update(1)
                if args.limit is not None and processed >= args.limit:
                    break

    token_counts_path = os.path.join(args.out_dir, "token_counts.csv")
    with open(token_counts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "token", "count"])
        for token_id, count in token_counter.most_common():
            writer.writerow([token_id, tokenizer.convert_ids_to_tokens(token_id), count])

    top_tokens_path = os.path.join(args.out_dir, f"top_{args.top_k}_tokens.txt")
    with open(top_tokens_path, "w", encoding="utf-8") as f:
        for rank, (token_id, count) in enumerate(token_counter.most_common(args.top_k), start=1):
            f.write(f"{rank:5d}  id={token_id:<8d} count={count:<12d} token={tokenizer.convert_ids_to_tokens(token_id)!r}\n")

    lengths = sorted(length_counter)

    def percentile(p):
        if not lengths:
            return 0
        idx = min(len(lengths) - 1, int(round((len(lengths) - 1) * p / 100.0)))
        return lengths[idx]

    special_token_counts = {
        str(token_id): {"token": tokenizer.convert_ids_to_tokens(token_id), "count": token_counter.get(token_id, 0)}
        for token_id in sorted(set(tokenizer.all_special_ids))
    }
    summary = {
        "data": args.data,
        "model": args.model,
        "rows_processed": processed,
        "total_message_tokens": sum(token_counter.values()),
        "vocab_size": len(tokenizer),
        "used_token_count": len(token_counter),
        "unused_token_count": len(tokenizer) - len(token_counter),
        "avg_tokens_per_row": (sum(token_counter.values()) / processed) if processed else 0,
        "min_tokens_per_row": lengths[0] if lengths else 0,
        "max_tokens_per_row": lengths[-1] if lengths else 0,
        "p50_tokens_per_row": percentile(50),
        "p90_tokens_per_row": percentile(90),
        "p95_tokens_per_row": percentile(95),
        "p99_tokens_per_row": percentile(99),
        "role_counts": dict(role_counter),
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
    for rank, (token_id, count) in enumerate(token_counter.most_common(min(args.top_k, 30)), start=1):
        print(f"{rank:5d}  id={token_id:<8d} count={count:<12d} token={tokenizer.convert_ids_to_tokens(token_id)!r}")


if __name__ == "__main__":
    main()
