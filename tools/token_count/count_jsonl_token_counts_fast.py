#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import Counter
from multiprocessing import Pool

from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_DATA = "/temp/csm/Dataset-train/mobimind_e2e_train.jsonl"
DEFAULT_MODEL = "/temp/zhangdelong/models/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40"

TOKENIZER = None
USE_CHAT_TEMPLATE = True


def init_worker(model, use_chat_template):
    global TOKENIZER, USE_CHAT_TEMPLATE
    TOKENIZER = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    USE_CHAT_TEMPLATE = use_chat_template


def encode_messages(messages):
    normalized = [{"role": item["role"], "content": item["content"]} for item in messages]
    if USE_CHAT_TEMPLATE:
        try:
            return TOKENIZER.apply_chat_template(normalized, tokenize=True, add_generation_prompt=False)
        except Exception:
            pass
    text = "\n".join(f"{m['role']}: {m['content']}" for m in normalized)
    return TOKENIZER.encode(text, add_special_tokens=True)


def process_lines(lines):
    counter = Counter()
    role_counts = Counter()
    rows = 0
    total_tokens = 0
    lengths = []
    for line in lines:
        if not line.strip():
            continue
        obj = json.loads(line)
        messages = obj["messages"]
        for message in messages:
            role_counts[message.get("role", "")] += 1
        token_ids = encode_messages(messages)
        counter.update(token_ids)
        total_tokens += len(token_ids)
        lengths.append(len(token_ids))
        rows += 1
    return dict(counter), dict(role_counts), rows, total_tokens, lengths


def iter_batches(path, batch_size, limit):
    batch = []
    seen = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            batch.append(line)
            seen += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []
            if limit is not None and seen >= limit:
                break
    if batch:
        yield batch


def percentile(sorted_values, p):
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * p / 100.0)))
    return sorted_values[idx]


def main():
    parser = argparse.ArgumentParser(description="Fast JSONL token frequency count for messages.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default="jsonl_token_count_outputs")
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    counter = Counter()
    role_counts = Counter()
    rows = 0
    total_tokens = 0
    lengths = []

    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.model, not args.no_chat_template),
    ) as pool:
        results = pool.imap_unordered(process_lines, iter_batches(args.data, args.batch_size, args.limit), chunksize=2)
        for batch_counter, batch_roles, batch_rows, batch_tokens, batch_lengths in tqdm(
            results, desc="counting token batches", unit="batch"
        ):
            counter.update(batch_counter)
            role_counts.update(batch_roles)
            rows += batch_rows
            total_tokens += batch_tokens
            lengths.extend(batch_lengths)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
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
            f.write(f"{rank:5d}  id={token_id:<8d} count={count:<12d} token={token!r}\n")

    unused_tokens_path = os.path.join(args.out_dir, "unused_tokens.csv")
    used = set(counter)
    with open(unused_tokens_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "token"])
        for token_id in range(len(tokenizer)):
            if token_id not in used:
                writer.writerow([token_id, tokenizer.convert_ids_to_tokens(token_id)])

    lengths.sort()
    special_counts = {
        str(token_id): {
            "token": tokenizer.convert_ids_to_tokens(token_id),
            "count": counter.get(token_id, 0),
        }
        for token_id in sorted(set(tokenizer.all_special_ids))
    }
    summary = {
        "data": args.data,
        "model": args.model,
        "rows_processed": rows,
        "total_message_tokens": total_tokens,
        "vocab_size": len(tokenizer),
        "used_token_count": len(counter),
        "unused_token_count": len(tokenizer) - len(counter),
        "avg_tokens_per_row": (total_tokens / rows) if rows else 0,
        "min_tokens_per_row": lengths[0] if lengths else 0,
        "max_tokens_per_row": lengths[-1] if lengths else 0,
        "p50_tokens_per_row": percentile(lengths, 50),
        "p90_tokens_per_row": percentile(lengths, 90),
        "p95_tokens_per_row": percentile(lengths, 95),
        "p99_tokens_per_row": percentile(lengths, 99),
        "role_counts": dict(role_counts),
        "special_token_counts": special_counts,
        "used_chat_template": not args.no_chat_template,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "token_counts_csv": token_counts_path,
        "unused_tokens_csv": unused_tokens_path,
        "top_tokens_txt": top_tokens_path,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTop tokens:")
    for rank, (token_id, count) in enumerate(counter.most_common(min(args.top_k, 30)), start=1):
        print(f"{rank:5d}  id={token_id:<8d} count={count:<12d} token={tokenizer.convert_ids_to_tokens(token_id)!r}")


if __name__ == "__main__":
    main()
