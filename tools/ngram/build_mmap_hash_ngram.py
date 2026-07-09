#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import struct
from collections import defaultdict
from pathlib import Path


MAGIC = b"MNNNGRM3"
VERSION = 3
FLAGS_TOP1 = 1
HASH_SEED = 1469598103934665603
HASH_PRIME = 1099511628211
HEADER_STRUCT = struct.Struct("<8sIIIIIIII")
SECTION_STRUCT = struct.Struct("<IIIIQQII")


def open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def align(value, alignment=64):
    return (value + alignment - 1) // alignment * alignment


def next_power_of_two(value):
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def hash_ngram(keys):
    h = HASH_SEED
    for token in keys:
        h ^= token & 0xFFFFFFFF
        h = (h * HASH_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def parse_tsv(path, max_key_len):
    sections = defaultdict(dict)
    input_rows = 0
    parsed_rows = 0
    duplicate_rows = 0
    with open_text(path) as f:
        for line in f:
            input_rows += 1
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("n\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            n = int(parts[0])
            if n <= 0 or n > max_key_len:
                continue
            key = tuple(int(x) for x in parts[1].split(",") if x)
            if len(key) != n:
                continue
            next_token = int(parts[2])
            parsed_rows += 1
            old = sections[n].get(key)
            if old is not None and old != next_token:
                duplicate_rows += 1
                continue
            sections[n][key] = next_token
    return sections, {
        "input_rows": input_rows,
        "parsed_rows": parsed_rows,
        "duplicate_rows_dropped": duplicate_rows,
    }


def build_section(n, mapping, load_factor):
    entries = [(key, next_token) for key, next_token in mapping.items()]
    bucket_count = next_power_of_two(max(1, math.ceil(len(entries) * load_factor)))
    buckets = [[] for _ in range(bucket_count)]
    for key, next_token in entries:
        bucket = hash_ngram(key) & (bucket_count - 1)
        buckets[bucket].append((key, next_token))

    offsets = [0]
    flat = []
    max_bucket = 0
    non_empty = 0
    for bucket_entries in buckets:
        bucket_entries.sort(key=lambda item: item[0])
        if bucket_entries:
            non_empty += 1
            max_bucket = max(max_bucket, len(bucket_entries))
        flat.extend(bucket_entries)
        offsets.append(len(flat))

    entry_struct = struct.Struct("<" + "i" * (n + 1))
    offset_blob = struct.pack("<" + "I" * len(offsets), *offsets)
    entry_blob = bytearray(entry_struct.size * len(flat))
    pos = 0
    for key, next_token in flat:
        entry_struct.pack_into(entry_blob, pos, *(key + (next_token,)))
        pos += entry_struct.size

    return {
        "n": n,
        "entry_count": len(flat),
        "bucket_count": bucket_count,
        "entry_stride": entry_struct.size,
        "offset_blob": offset_blob,
        "entry_blob": bytes(entry_blob),
        "max_bucket_size": max_bucket,
        "non_empty_buckets": non_empty,
    }


def main():
    parser = argparse.ArgumentParser(description="Build mmap-direct hash-index ngram binary for top1 lookahead.")
    parser.add_argument("--input", required=True, help="Input top1 ngram TSV/TSV.GZ")
    parser.add_argument("--output", required=True, help="Output .mnnngram3 file")
    parser.add_argument("--max-key-len", type=int, default=4)
    parser.add_argument("--load-factor", type=float, default=2.0, help="bucket_count ~= next_power_of_two(entries * load_factor)")
    args = parser.parse_args()

    if args.max_key_len <= 0:
        raise ValueError("--max-key-len must be positive")
    if args.load_factor < 1.0:
        raise ValueError("--load-factor must be >= 1.0")

    parsed, stats = parse_tsv(args.input, args.max_key_len)
    sections = [build_section(n, parsed[n], args.load_factor) for n in sorted(parsed)]

    section_count = len(sections)
    desc_offset = HEADER_STRUCT.size
    data_offset = align(desc_offset + section_count * SECTION_STRUCT.size)

    cursor = data_offset
    descs = []
    for sec in sections:
        bucket_offsets_offset = cursor
        cursor += len(sec["offset_blob"])
        cursor = align(cursor)
        entries_offset = cursor
        cursor += len(sec["entry_blob"])
        cursor = align(cursor)
        descs.append((bucket_offsets_offset, entries_offset))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        header = HEADER_STRUCT.pack(
            MAGIC,
            VERSION,
            FLAGS_TOP1,
            args.max_key_len,
            section_count,
            desc_offset,
            data_offset,
            HASH_SEED & 0xFFFFFFFF,
            0,
        )
        f.write(header)
        for sec, (bucket_offsets_offset, entries_offset) in zip(sections, descs):
            f.write(SECTION_STRUCT.pack(
                sec["n"],
                sec["entry_count"],
                sec["bucket_count"],
                sec["entry_stride"],
                bucket_offsets_offset,
                entries_offset,
                0,
                0,
            ))
        f.seek(data_offset)
        for sec, (bucket_offsets_offset, entries_offset) in zip(sections, descs):
            f.seek(bucket_offsets_offset)
            f.write(sec["offset_blob"])
            f.seek(entries_offset)
            f.write(sec["entry_blob"])

    summary = {
        "input": args.input,
        "output": str(out),
        "bytes": out.stat().st_size,
        "version": VERSION,
        "flags": FLAGS_TOP1,
        "max_key_len": args.max_key_len,
        "load_factor": args.load_factor,
        **stats,
        "sections": [
            {
                "n": sec["n"],
                "entries": sec["entry_count"],
                "bucket_count": sec["bucket_count"],
                "entry_stride": sec["entry_stride"],
                "max_bucket_size": sec["max_bucket_size"],
                "non_empty_buckets": sec["non_empty_buckets"],
            }
            for sec in sections
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
