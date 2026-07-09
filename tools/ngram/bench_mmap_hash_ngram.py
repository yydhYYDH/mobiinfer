#!/usr/bin/env python3
import argparse
import gzip
import json
import mmap
import random
import struct
import time
from pathlib import Path


MAGIC = b"MNNNGRM3"
HASH_SEED = 1469598103934665603
HASH_PRIME = 1099511628211
HEADER_STRUCT = struct.Struct("<8sIIIIIIII")
SECTION_STRUCT = struct.Struct("<IIIIQQII")


def open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def hash_ngram(keys):
    h = HASH_SEED
    for token in keys:
        h ^= token & 0xFFFFFFFF
        h = (h * HASH_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


class MmapHashNgram:
    def __init__(self, path):
        start = time.perf_counter()
        self.file = Path(path).open("rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.sections = {}
        self._parse()
        self.open_seconds = time.perf_counter() - start

    def close(self):
        self.mm.close()
        self.file.close()

    def _parse(self):
        magic, version, flags, max_key_len, section_count, desc_offset, data_offset, hash_seed_low, reserved = HEADER_STRUCT.unpack_from(self.mm, 0)
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        if version != 3:
            raise ValueError(f"bad version: {version}")
        for i in range(section_count):
            pos = desc_offset + i * SECTION_STRUCT.size
            n, entry_count, bucket_count, entry_stride, bucket_offsets_offset, entries_offset, _, _ = SECTION_STRUCT.unpack_from(self.mm, pos)
            self.sections[n] = {
                "n": n,
                "entry_count": entry_count,
                "bucket_count": bucket_count,
                "entry_stride": entry_stride,
                "bucket_offsets_offset": bucket_offsets_offset,
                "entries_offset": entries_offset,
                "entry_struct": struct.Struct("<" + "i" * (n + 1)),
            }

    def lookup(self, keys):
        n = len(keys)
        sec = self.sections.get(n)
        if sec is None:
            return None
        bucket = hash_ngram(keys) & (sec["bucket_count"] - 1)
        offsets_pos = sec["bucket_offsets_offset"] + bucket * 4
        begin = struct.unpack_from("<I", self.mm, offsets_pos)[0]
        end = struct.unpack_from("<I", self.mm, offsets_pos + 4)[0]
        entry_struct = sec["entry_struct"]
        entries_offset = sec["entries_offset"]
        stride = sec["entry_stride"]
        for i in range(begin, end):
            row = entry_struct.unpack_from(self.mm, entries_offset + i * stride)
            if row[:-1] == keys:
                return row[-1]
        return None


def load_queries(path, max_key_len, limit):
    queries = []
    with open_text(path) as f:
        for line in f:
            if not line or line.startswith("n\t") or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            n = int(parts[0])
            if n <= 0 or n > max_key_len:
                continue
            key = tuple(int(x) for x in parts[1].split(",") if x)
            if len(key) != n:
                continue
            queries.append((key, int(parts[2])))
            if limit and len(queries) >= limit:
                break
    return queries


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, int(round((len(values) - 1) * p)))
    return values[idx]


def main():
    parser = argparse.ArgumentParser(description="Benchmark mmap-direct hash-index ngram lookup.")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--queries", required=True, help="TSV/TSV.GZ used to sample hit queries")
    parser.add_argument("--max-key-len", type=int, default=4)
    parser.add_argument("--query-limit", type=int, default=200000)
    parser.add_argument("--lookups", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    queries = load_queries(args.queries, args.max_key_len, args.query_limit)
    if not queries:
        raise SystemExit("no queries loaded")
    random.shuffle(queries)
    keys = [item[0] for item in queries]
    expected = {item[0]: item[1] for item in queries}

    table = MmapHashNgram(args.binary)
    try:
        warmup = min(10000, len(keys))
        for key in keys[:warmup]:
            table.lookup(key)

        lookup_count = args.lookups
        start = time.perf_counter()
        hit = 0
        checksum = 0
        for i in range(lookup_count):
            key = keys[i % len(keys)]
            value = table.lookup(key)
            if value is not None:
                hit += 1
                checksum ^= value
        elapsed = time.perf_counter() - start

        sample_times = []
        sample_count = min(20000, len(keys))
        for key in keys[:sample_count]:
            t0 = time.perf_counter_ns()
            table.lookup(key)
            sample_times.append(time.perf_counter_ns() - t0)

        miss_keys = []
        for key in keys[:sample_count]:
            miss = list(key)
            miss[-1] = miss[-1] + 1000000007
            miss_keys.append(tuple(miss))
        miss_start = time.perf_counter()
        miss_hit = 0
        for key in miss_keys:
            if table.lookup(key) is not None:
                miss_hit += 1
        miss_elapsed = time.perf_counter() - miss_start

        result = {
            "binary": args.binary,
            "queries": args.queries,
            "loaded_queries": len(keys),
            "mmap_open_and_parse_ms": table.open_seconds * 1000,
            "hit_lookups": lookup_count,
            "hit_count": hit,
            "hit_elapsed_s": elapsed,
            "hit_lookup_per_s": lookup_count / elapsed,
            "hit_avg_ns": elapsed * 1e9 / lookup_count,
            "sample_p50_ns": percentile(sample_times, 0.50),
            "sample_p90_ns": percentile(sample_times, 0.90),
            "sample_p99_ns": percentile(sample_times, 0.99),
            "miss_lookups": len(miss_keys),
            "miss_false_hits": miss_hit,
            "miss_elapsed_s": miss_elapsed,
            "miss_lookup_per_s": len(miss_keys) / miss_elapsed if miss_elapsed else None,
            "checksum": checksum,
            "sections": {
                n: {
                    "entries": sec["entry_count"],
                    "bucket_count": sec["bucket_count"],
                    "entry_stride": sec["entry_stride"],
                }
                for n, sec in table.sections.items()
            },
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        table.close()


if __name__ == "__main__":
    main()
