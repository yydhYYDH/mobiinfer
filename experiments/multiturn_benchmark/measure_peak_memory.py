#!/usr/bin/env python3
"""Measure peak RSS, PSS, and USS for a child process on Linux or Android."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


STATUS_FIELDS = {
    "VmHWM": "vm_hwm_kb",
    "VmRSS": "vm_rss_kb",
    "VmPeak": "vm_peak_kb",
    "VmSize": "vm_size_kb",
    "VmSwap": "vm_swap_kb",
    "RssAnon": "rss_anon_kb",
    "RssFile": "rss_file_kb",
    "RssShmem": "rss_shmem_kb",
}

SMAPS_FIELDS = {
    "Rss": "smaps_rss_kb",
    "Pss": "pss_kb",
    "Pss_Anon": "pss_anon_kb",
    "Pss_File": "pss_file_kb",
    "Pss_Shmem": "pss_shmem_kb",
    "Private_Clean": "private_clean_kb",
    "Private_Dirty": "private_dirty_kb",
    "Private_Hugetlb": "private_hugetlb_kb",
    "Swap": "swap_kb",
    "SwapPss": "swap_pss_kb",
}


def parse_proc_kb(path: Path, fields: dict[str, str]) -> dict[str, int]:
    values: dict[str, int] = {}
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        name, separator, rest = line.partition(":")
        if not separator or name not in fields:
            continue
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            values[fields[name]] = int(parts[0])
        except ValueError:
            continue
    return values


def sample_process(pid: int, elapsed_s: float) -> dict[str, float | int]:
    proc = Path("/proc") / str(pid)
    sample: dict[str, float | int] = {"elapsed_s": elapsed_s}
    sample.update(parse_proc_kb(proc / "status", STATUS_FIELDS))
    try:
        sample.update(parse_proc_kb(proc / "smaps_rollup", SMAPS_FIELDS))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    sample["uss_kb"] = sum(
        int(sample.get(key, 0))
        for key in ("private_clean_kb", "private_dirty_kb", "private_hugetlb_kb")
    )
    return sample


def update_peaks(peaks: dict[str, dict[str, float | int]], sample: dict[str, float | int]) -> None:
    for key, value in sample.items():
        if key == "elapsed_s" or not key.endswith("_kb"):
            continue
        current = peaks.get(key)
        if current is None or int(value) > int(current["value_kb"]):
            peaks[key] = {"value_kb": int(value), "elapsed_s": float(sample["elapsed_s"])}


def write_samples(path: Path, samples: list[dict[str, float | int]]) -> None:
    fieldnames = ["elapsed_s"] + sorted(
        {key for sample in samples for key in sample if key != "elapsed_s"}
    )
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def parse_env(items: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"Invalid --env value: {item!r}")
        env[key] = value
    return env


def mib(value_kb: int | None) -> float | None:
    return value_kb / 1024.0 if value_kb is not None else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--samples-csv", type=Path)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--sample-interval-ms", type=float, default=100.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if args.sample_interval_ms <= 0:
        parser.error("--sample-interval-ms must be positive")

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    log_path = (args.log or args.output.with_suffix(".log")).resolve()
    samples_path = (args.samples_csv or args.output.with_suffix(".samples.csv")).resolve()
    if log_path.exists():
        raise FileExistsError(log_path)
    if samples_path.exists():
        raise FileExistsError(samples_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.parent.mkdir(parents=True, exist_ok=True)

    env = parse_env(args.env)
    samples: list[dict[str, float | int]] = []
    peaks: dict[str, dict[str, float | int]] = {}
    started = time.monotonic()
    timed_out = False

    with log_path.open("x", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=args.cwd,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        deadline = started + args.timeout_seconds
        while True:
            elapsed = time.monotonic() - started
            try:
                sample = sample_process(process.pid, elapsed)
                samples.append(sample)
                update_peaks(peaks, sample)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass

            return_code = process.poll()
            if return_code is not None:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                process.terminate()
                try:
                    return_code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    return_code = process.wait()
                break
            time.sleep(args.sample_interval_ms / 1000.0)

    wall_time_s = time.monotonic() - started
    write_samples(samples_path, samples)

    def peak_kb(key: str) -> int | None:
        item = peaks.get(key)
        return int(item["value_kb"]) if item else None

    result = {
        "command": command,
        "cwd": str(args.cwd.resolve()) if args.cwd else None,
        "pid": process.pid,
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_time_s": wall_time_s,
        "sample_interval_ms": args.sample_interval_ms,
        "sample_count": len(samples),
        "platform": platform.platform(),
        "memory_unit": "MiB (KiB / 1024)",
        "peak_rss_mb": mib(peak_kb("vm_hwm_kb")),
        "peak_sampled_rss_mb": mib(peak_kb("smaps_rss_kb")),
        "peak_pss_mb": mib(peak_kb("pss_kb")),
        "peak_uss_mb": mib(peak_kb("uss_kb")),
        "peak_swap_pss_mb": mib(peak_kb("swap_pss_kb")),
        "peak_vm_size_mb": mib(peak_kb("vm_peak_kb")),
        "peak_details": peaks,
        "log": str(log_path),
        "samples_csv": str(samples_path),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.quiet:
        def display(value: float | None) -> str:
            return f"{value:.2f}" if value is not None else "NA"

        print(
            f"return_code={return_code} peak_rss_mb={display(result['peak_rss_mb'])} "
            f"peak_pss_mb={display(result['peak_pss_mb'])} "
            f"peak_uss_mb={display(result['peak_uss_mb'])}"
        )
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 124 if timed_out else int(return_code)


if __name__ == "__main__":
    sys.exit(main())
