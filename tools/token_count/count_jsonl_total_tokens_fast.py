#!/usr/bin/env python3
import argparse
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
    total_tokens = 0
    lengths = []
    role_counts = Counter()
    special_counts = Counter()
    special_ids = set(TOKENIZER.all_special_ids)
    for line in lines:
        if not line.strip():
            continue
        obj = json.loads(line)
        messages = obj["messages"]
        for message in messages:
            role_counts[message.get("role", "")] += 1
        token_ids = encode_messages(messages)
        length = len(token_ids)
        total_tokens += length
        lengths.append(length)
        for token_id in token_ids:
            if token_id in special_ids:
                special_counts[token_id] += 1
    return total_tokens, lengths, dict(role_counts), dict(special_counts)


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
    parser = argparse.ArgumentParser(description="Fast total token count for JSONL messages.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default="jsonl_total_outputs")
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    total_tokens = 0
    lengths = []
    role_counts = Counter()
    special_counts = Counter()
    rows = 0

    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.model, not args.no_chat_template),
    ) as pool:
        results = pool.imap_unordered(process_lines, iter_batches(args.data, args.batch_size, args.limit), chunksize=2)
        for batch_tokens, batch_lengths, batch_roles, batch_specials in tqdm(results, desc="tokenizing batches", unit="batch"):
            total_tokens += batch_tokens
            lengths.extend(batch_lengths)
            rows += len(batch_lengths)
            role_counts.update(batch_roles)
            special_counts.update(batch_specials)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    lengths.sort()
    summary = {
        "data": args.data,
        "model": args.model,
        "rows_processed": rows,
        "total_message_tokens": total_tokens,
        "avg_tokens_per_row": (total_tokens / rows) if rows else 0,
        "min_tokens_per_row": lengths[0] if lengths else 0,
        "max_tokens_per_row": lengths[-1] if lengths else 0,
        "p50_tokens_per_row": percentile(lengths, 50),
        "p90_tokens_per_row": percentile(lengths, 90),
        "p95_tokens_per_row": percentile(lengths, 95),
        "p99_tokens_per_row": percentile(lengths, 99),
        "role_counts": dict(role_counts),
        "special_token_counts": {
            str(token_id): {
                "token": tokenizer.convert_ids_to_tokens(token_id),
                "count": special_counts.get(token_id, 0),
            }
            for token_id in sorted(set(tokenizer.all_special_ids))
        },
        "used_chat_template": not args.no_chat_template,
        "workers": args.workers,
        "batch_size": args.batch_size,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
