#!/usr/bin/env python3
import argparse
import gzip
import struct
from pathlib import Path

MAGIC = b"MNNNGRM2"
VERSION = 2
MAX_KEY_LEN = 8


def open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def parse_line(line, max_key_len):
    line = line.rstrip("\n")
    if not line or line.startswith("#") or line.startswith("n\t"):
        return None
    parts = line.split("\t")
    if len(parts) < 4:
        return None
    key_len = int(parts[0])
    if key_len <= 0 or key_len > max_key_len or key_len > MAX_KEY_LEN:
        return None
    keys = [int(x) for x in parts[1].split(",") if x]
    if len(keys) != key_len:
        return None
    keys.extend([-1] * (max_key_len - key_len))
    next_token = int(parts[2])
    count = int(parts[3])
    if count <= 0:
        return None
    return key_len, keys, next_token, count


def main():
    parser = argparse.ArgumentParser(description="Convert MNN lookahead ngram TSV to mmap-friendly binary format.")
    parser.add_argument("--input", required=True, help="Input TSV or TSV.GZ file")
    parser.add_argument("--output", required=True, help="Output .mnnngram/.bin file")
    parser.add_argument("--max-key-len", type=int, default=8, help="Keep entries with n <= max-key-len")
    args = parser.parse_args()
    if args.max_key_len <= 0 or args.max_key_len > MAX_KEY_LEN:
        raise ValueError(f"--max-key-len must be in [1, {MAX_KEY_LEN}]")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    entry_count = 0
    with open_text(args.input) as src, out.open("wb") as dst:
        dst.write(MAGIC)
        dst.write(struct.pack("<IIII", VERSION, args.max_key_len, 0, 0))
        for line in src:
            entry = parse_line(line, args.max_key_len)
            if entry is None:
                continue
            key_len, keys, next_token, count = entry
            dst.write(struct.pack("<I", key_len))
            dst.write(struct.pack(f"<{args.max_key_len}i", *keys))
            dst.write(struct.pack("<ii", next_token, count))
            entry_count += 1
        dst.seek(len(MAGIC))
        dst.write(struct.pack("<IIII", VERSION, args.max_key_len, 0, entry_count))

    print({
        "input": str(args.input),
        "output": str(out),
        "entries": entry_count,
        "bytes": out.stat().st_size,
        "max_key_len": args.max_key_len,
    })


if __name__ == "__main__":
    main()
