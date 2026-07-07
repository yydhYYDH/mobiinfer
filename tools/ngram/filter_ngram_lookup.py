#!/usr/bin/env python3
import gzip
import json
import os
import time
from collections import defaultdict

IN_PATH = "/home/ma-user/workspace/csm/mobiinfer/ngram/output/assistant_ngram_lookup_counts.tsv.gz"
OUTDIR = "/home/ma-user/workspace/csm/mobiinfer/ngram/output"
MIN_COUNT = 20
TOP_KS = (1, 3)

os.makedirs(OUTDIR, exist_ok=True)
start = time.time()

out_files = {}
for k in TOP_KS:
    path = os.path.join(OUTDIR, f"assistant_ngram_min20_top{k}.tsv.gz")
    f = gzip.open(path, "wt", encoding="utf-8")
    f.write("n\tkey_token_ids\tnext_token_id\tcount\n")
    out_files[k] = (path, f)

summary = {
    "source": IN_PATH,
    "min_count": MIN_COUNT,
    "top_ks": list(TOP_KS),
    "ngram": {str(n): {f"top{k}": {"pairs": 0, "keys": 0, "occurrences": 0} for k in TOP_KS} for n in range(1, 9)},
}

current_key = None
items = []
raw_lines = 0

def flush_group(key, group_items):
    if key is None:
        return
    n, key_tokens = key
    filtered = [item for item in group_items if item[1] >= MIN_COUNT]
    if not filtered:
        return
    filtered.sort(key=lambda item: (-item[1], item[0]))
    for top_k in TOP_KS:
        selected = filtered[:top_k]
        if not selected:
            continue
        path, f = out_files[top_k]
        for next_token, count in selected:
            f.write(f"{n}\t{key_tokens}\t{next_token}\t{count}\n")
        stat = summary["ngram"][str(n)][f"top{top_k}"]
        stat["keys"] += 1
        stat["pairs"] += len(selected)
        stat["occurrences"] += sum(count for _, count in selected)

with gzip.open(IN_PATH, "rt", encoding="utf-8") as source:
    header = next(source)
    for line in source:
        raw_lines += 1
        n, key_tokens, next_token, count = line.rstrip("\n").split("\t")
        key = (int(n), key_tokens)
        if current_key is not None and key != current_key:
            flush_group(current_key, items)
            items = []
        current_key = key
        items.append((int(next_token), int(count)))
        if raw_lines % 2000000 == 0:
            print(f"processed {raw_lines} lines elapsed={time.time()-start:.1f}s", flush=True)
flush_group(current_key, items)

for path, f in out_files.values():
    f.close()
summary["elapsed_sec"] = time.time() - start
summary["outputs"] = {f"top{k}": out_files[k][0] for k in TOP_KS}
summary_path = os.path.join(OUTDIR, "assistant_ngram_min20_top_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
