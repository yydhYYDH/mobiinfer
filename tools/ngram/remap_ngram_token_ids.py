#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def open_text_read(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def open_text_write(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt")
    return path.open("w")


def load_mapping(path):
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        return {int(old): int(new) for old, new in raw.items() if new is not None}
    if isinstance(raw, list):
        return {old: int(new) for old, new in enumerate(raw) if new is not None and int(new) >= 0}
    raise TypeError("mapping must be a dict {old_id: new_id} or a list indexed by old_id")


def parse_tsv_line(line, max_key_len):
    line = line.rstrip("\n")
    if not line or line.startswith("#") or line.startswith("n\t"):
        return None
    parts = line.split("\t")
    if len(parts) < 4:
        return None
    n = int(parts[0])
    if n <= 0 or n > max_key_len:
        return None
    key = tuple(int(x) for x in parts[1].split(",") if x)
    if len(key) != n:
        return None
    next_token = int(parts[2])
    count = int(parts[3])
    if count <= 0:
        return None
    return n, key, next_token, count


def remap_entry(entry, mapping):
    n, key, next_token, count = entry
    try:
        new_key = tuple(mapping[token] for token in key)
        new_next = mapping[next_token]
    except KeyError:
        return None
    return n, new_key, new_next, count


def main():
    parser = argparse.ArgumentParser(description="Remap lookahead ngram TSV token ids with old_to_new_token_id.json.")
    parser.add_argument("--input", required=True, help="Input ngram TSV or TSV.GZ in n/key_token_ids/next_token_id/count format")
    parser.add_argument("--mapping", required=True, help="JSON mapping from old token id to new token id")
    parser.add_argument("--output", required=True, help="Output remapped TSV or TSV.GZ")
    parser.add_argument("--max-key-len", type=int, default=8, help="Drop input rows with n greater than this")
    parser.add_argument("--min-count", type=int, default=1, help="Drop remapped candidates with count below this")
    parser.add_argument("--top-k", type=int, default=0, help="Keep top K next tokens per key by count; 0 keeps all")
    args = parser.parse_args()

    if args.max_key_len <= 0:
        raise ValueError("--max-key-len must be positive")
    if args.min_count <= 0:
        raise ValueError("--min-count must be positive")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")

    mapping = load_mapping(args.mapping)
    grouped = defaultdict(lambda: defaultdict(int))
    input_rows = 0
    parsed_rows = 0
    kept_rows_before_filter = 0
    dropped_unmapped = 0

    with open_text_read(args.input) as src:
        for line in src:
            input_rows += 1
            entry = parse_tsv_line(line, args.max_key_len)
            if entry is None:
                continue
            parsed_rows += 1
            remapped = remap_entry(entry, mapping)
            if remapped is None:
                dropped_unmapped += 1
                continue
            n, key, next_token, count = remapped
            grouped[(n, key)][next_token] += count
            kept_rows_before_filter += 1

    output_rows = 0
    candidate_rows_after_merge = 0
    with open_text_write(args.output) as dst:
        dst.write("n\tkey_token_ids\tnext_token_id\tcount\n")
        for n, key in sorted(grouped.keys(), key=lambda item: (item[0], item[1])):
            candidates = grouped[(n, key)]
            ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
            candidate_rows_after_merge += len(ranked)
            if args.top_k:
                ranked = ranked[: args.top_k]
            for next_token, count in ranked:
                if count < args.min_count:
                    continue
                dst.write(f"{n}\t{','.join(str(x) for x in key)}\t{next_token}\t{count}\n")
                output_rows += 1

    out = Path(args.output)
    print(json.dumps({
        "input": args.input,
        "mapping": args.mapping,
        "output": str(out),
        "mapping_size": len(mapping),
        "input_rows": input_rows,
        "parsed_rows": parsed_rows,
        "remapped_rows_before_merge": kept_rows_before_filter,
        "dropped_unmapped_rows": dropped_unmapped,
        "unique_keys": len(grouped),
        "candidate_rows_after_merge": candidate_rows_after_merge,
        "output_rows": output_rows,
        "output_bytes": out.stat().st_size,
        "max_key_len": args.max_key_len,
        "min_count": args.min_count,
        "top_k": args.top_k,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
