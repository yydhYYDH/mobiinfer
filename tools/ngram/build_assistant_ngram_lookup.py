#!/usr/bin/env python3
import gzip
import json
import os
import statistics
import time
from collections import Counter, defaultdict

from transformers import AutoTokenizer


DEFAULT_DATA = "/temp/csm/Dataset-train/mobimind_e2e_train.jsonl"
DEFAULT_MODEL = "/temp/zhangdelong/models/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40"
DEFAULT_OUTDIR = "/home/ma-user/workspace/csm/mobiinfer/ngram/output"
DEFAULT_LOGDIR = "/home/ma-user/workspace/csm/mobiinfer/ngram/logs"
MAX_N = 8
TOPK = 8
REPORT_EVERY = 200000


def main():
    os.makedirs(DEFAULT_OUTDIR, exist_ok=True)
    os.makedirs(DEFAULT_LOGDIR, exist_ok=True)
    log_path = os.path.join(DEFAULT_LOGDIR, "assistant_ngram_build.log")
    start = time.time()

    def log(message):
        line = f"[{time.strftime('%F %T')}] {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")

    texts = Counter()
    role_counts = Counter()
    rows = 0
    bad = 0
    assistants = 0

    log("pass1 start counting unique assistant outputs")
    with open(DEFAULT_DATA, "r", encoding="utf-8") as data_file:
        for line in data_file:
            rows += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue
            for message in obj.get("messages", []):
                role = message.get("role")
                role_counts[role] += 1
                if role == "assistant":
                    assistants += 1
                    texts[message.get("content", "")] += 1
            if rows % REPORT_EVERY == 0:
                log(
                    "pass1 rows=%d assistants=%d unique_assistant=%d elapsed=%.1fs"
                    % (rows, assistants, len(texts), time.time() - start)
                )
    log(
        "pass1 done rows=%d bad=%d assistants=%d unique_assistant=%d elapsed=%.1fs roles=%s"
        % (rows, bad, assistants, len(texts), time.time() - start, dict(role_counts))
    )

    log("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True, local_files_only=True)

    tables = [None] + [defaultdict(Counter) for _ in range(MAX_N)]
    len_hist = Counter()
    token_total = 0
    unique_items = list(texts.items())

    log("pass2 start tokenizing unique assistant outputs and accumulating weighted ngrams")
    for processed, (text, freq) in enumerate(unique_items, 1):
        ids = tokenizer.encode(text, add_special_tokens=False)
        length = len(ids)
        len_hist[length] += freq
        token_total += length * freq
        for n in range(1, MAX_N + 1):
            if length <= n:
                continue
            table = tables[n]
            for i in range(length - n):
                table[tuple(ids[i : i + n])][ids[i + n]] += freq
        if processed % 50000 == 0:
            log(
                "pass2 unique_processed=%d/%d elapsed=%.1fs keys=%s"
                % (processed, len(unique_items), time.time() - start, [len(tables[n]) for n in range(1, MAX_N + 1)])
            )
    log("pass2 done elapsed=%.1fs" % (time.time() - start))

    summary = {
        "data": DEFAULT_DATA,
        "model": DEFAULT_MODEL,
        "scope": "assistant messages only",
        "rows": rows,
        "bad_json_rows": bad,
        "assistant_messages": assistants,
        "unique_assistant_outputs": len(texts),
        "total_assistant_tokens": token_total,
        "avg_assistant_tokens": token_total / assistants if assistants else 0,
        "max_n": MAX_N,
        "ngram": {},
    }
    if len_hist:
        summary["token_len_min"] = min(len_hist)
        summary["token_len_p50"] = percentile_from_hist(len_hist, assistants, 0.5)
        summary["token_len_p95"] = percentile_from_hist(len_hist, assistants, 0.95)
        summary["token_len_max"] = max(len_hist)

    log("writing lookup table files")
    counts_path = os.path.join(DEFAULT_OUTDIR, "assistant_ngram_lookup_counts.tsv.gz")
    top_path = os.path.join(DEFAULT_OUTDIR, "assistant_ngram_lookup_top.jsonl.gz")
    examples_path = os.path.join(DEFAULT_OUTDIR, "assistant_ngram_examples.txt")

    with gzip.open(counts_path, "wt", encoding="utf-8") as counts_file, gzip.open(
        top_path, "wt", encoding="utf-8"
    ) as top_file:
        counts_file.write("n\tkey_token_ids\tnext_token_id\tcount\n")
        for n in range(1, MAX_N + 1):
            table = tables[n]
            key_next_pairs = 0
            total_occurrences = 0
            max_branch = 0
            for key, counter in table.items():
                max_branch = max(max_branch, len(counter))
                key_str = ",".join(map(str, key))
                items = counter.most_common()
                key_next_pairs += len(items)
                total = sum(counter.values())
                total_occurrences += total
                for next_token, count in items:
                    counts_file.write(f"{n}\t{key_str}\t{next_token}\t{count}\n")
                top_file.write(
                    json.dumps(
                        {
                            "n": n,
                            "key": list(key),
                            "key_text": tokenizer.decode(list(key)),
                            "total": total,
                            "next": [
                                {"token": token, "count": count, "text": tokenizer.decode([token])}
                                for token, count in items[:TOPK]
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            summary["ngram"][str(n)] = {
                "unique_keys": len(table),
                "unique_key_next_pairs": key_next_pairs,
                "total_occurrences": total_occurrences,
                "max_successors_per_key": max_branch,
            }
            log(f"wrote n={n} keys={len(table)} pairs={key_next_pairs} occ={total_occurrences}")

    with open(examples_path, "w", encoding="utf-8") as examples_file:
        for n in range(1, MAX_N + 1):
            examples_file.write(f"\n## n={n}\n")
            ranked = sorted(tables[n].items(), key=lambda item: sum(item[1].values()), reverse=True)[:30]
            for key, counter in ranked:
                examples_file.write(
                    json.dumps(
                        {
                            "key": list(key),
                            "key_text": tokenizer.decode(list(key)),
                            "total": sum(counter.values()),
                            "next": [
                                {"token": token, "text": tokenizer.decode([token]), "count": count}
                                for token, count in counter.most_common(5)
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary_path = os.path.join(DEFAULT_OUTDIR, "assistant_ngram_summary.json")
    summary["outputs"] = {
        "counts_tsv_gz": counts_path,
        "top_jsonl_gz": top_path,
        "examples_txt": examples_path,
        "log": log_path,
    }
    summary["elapsed_sec"] = time.time() - start
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
    log(f"done summary={summary_path} elapsed={summary['elapsed_sec']:.1f}s")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def percentile_from_hist(hist, total, ratio):
    threshold = total * ratio
    cumulative = 0
    for value in sorted(hist):
        cumulative += hist[value]
        if cumulative >= threshold:
            return value
    return max(hist)


if __name__ == "__main__":
    main()
