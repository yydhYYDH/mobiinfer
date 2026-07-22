#!/usr/bin/env python3
"""Summarize raw multi-turn phone benchmark logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore") if path.exists() else ""

    def get_float(pattern: str):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    def get_int(pattern: str):
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None

    return {
        "ok": "#################################" in text,
        "prompt_tokens": get_int(r"prompt tokens num = (\d+)"),
        "decode_tokens": get_int(r"decode tokens num = (\d+)"),
        "vision_time_s": get_float(r"vision time = ([0-9.]+) s"),
        "prefill_time_s": get_float(r"prefill time = ([0-9.]+) s"),
        "decode_time_s": get_float(r"decode time = ([0-9.]+) s"),
        "prefill_speed_tok_s": get_float(r"prefill speed = ([0-9.]+) tok/s"),
        "decode_speed_tok_s": get_float(r"decode speed = ([0-9.]+) tok/s"),
    }


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]

    def total(key: str):
        values = [r[key] for r in ok if r[key] is not None]
        return sum(values) if values else None

    def avg(key: str):
        values = [r[key] for r in ok if r[key] is not None]
        return sum(values) / len(values) if values else None

    prefill_tokens = total("prompt_tokens")
    decode_tokens = total("decode_tokens")
    prefill_time = total("prefill_time_s")
    decode_time = total("decode_time_s")
    return {
        "count": len(ok),
        "total": len(rows),
        "avg_prompt_tokens": avg("prompt_tokens"),
        "avg_decode_tokens": avg("decode_tokens"),
        "total_prefill_time_s": prefill_time,
        "total_decode_time_s": decode_time,
        "total_vision_time_s": total("vision_time_s"),
        "effective_prefill_speed_tok_s": (
            prefill_tokens / prefill_time if prefill_tokens and prefill_time else None
        ),
        "aggregate_decode_speed_tok_s": (
            decode_tokens / decode_time if decode_tokens and decode_time else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    args = parser.parse_args()

    bench = Path(args.bench)
    manifest = bench / "manifest.tsv"
    rows = []
    with manifest.open("r", encoding="utf-8") as f:
        for item in csv.DictReader(f, delimiter="\t"):
            case = f"traj_{item['trajectory_id']}_step_{item['step_id']}"
            row = {
                "trajectory_id": item["trajectory_id"],
                "step_id": item["step_id"],
                "history_count": int(item["history_count"]),
                "task": item["task"],
            }
            for variant, logdir in [
                ("baseline", bench / "logs_raw_baseline"),
                ("mobiinfer", bench / "logs_raw_mobiinfer"),
            ]:
                parsed = parse_log(logdir / f"{case}.log")
                rows.append({"variant": variant, **row, **parsed})

    by_variant = {}
    for variant in ["baseline", "mobiinfer"]:
        by_variant[variant] = summarize([r for r in rows if r["variant"] == variant])

    base = by_variant["baseline"]
    mobi = by_variant["mobiinfer"]
    speedups = {}
    for key in ["total_prefill_time_s", "total_decode_time_s", "total_vision_time_s"]:
        if base.get(key) and mobi.get(key):
            speedups[key.replace("total_", "").replace("_time_s", "_speedup")] = base[key] / mobi[key]
    if base.get("total_prefill_time_s") and base.get("total_decode_time_s") and mobi.get("total_prefill_time_s") and mobi.get("total_decode_time_s"):
        base_total = base["total_prefill_time_s"] + base["total_decode_time_s"] + (base.get("total_vision_time_s") or 0)
        mobi_total = mobi["total_prefill_time_s"] + mobi["total_decode_time_s"] + (mobi.get("total_vision_time_s") or 0)
        speedups["e2e_speedup"] = base_total / mobi_total

    summary = {"variants": by_variant, "speedups": speedups, "rows": rows}
    out = bench / "summary_multiturn_raw.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out), "speedups": speedups}, ensure_ascii=False))


if __name__ == "__main__":
    main()
