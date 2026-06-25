#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict


def f(row, key, default=0.0):
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def s(row, key, default=""):
    value = row.get(key, "")
    return default if value is None else value


def add_stat(stats, key, row):
    item = stats[key]
    item["time"] += f(row, "total_ms", f(row, "avg_ms"))
    item["dispatch"] += f(row, "dispatch_ms")
    item["wait"] += f(row, "wait_ms")
    item["io"] += f(row, "io_bytes")
    item["static"] += f(row, "static_bytes")
    item["flops"] += f(row, "flopsM")
    item["count"] += int(f(row, "count", f(row, "called_times", 1.0)))


def gbps(bytes_value, ms):
    if ms <= 0:
        return 0.0
    return bytes_value / ms / 1_000_000.0


def pct(value, total):
    if total <= 0:
        return 0.0
    return value * 100.0 / total


def classify(row):
    op_type = s(row, "op_type")
    phase = s(row, "phase", "unknown")
    avg = f(row, "avg_ms")
    wait = f(row, "wait_ms")
    gflops = f(row, "gflops")
    total_bw = f(row, "total_bw_gbps")
    m = f(row, "gemm_m")
    weight_bits = s(row, "weight_bits")
    kernel = s(row, "kernel")

    reasons = []
    if wait > max(0.1, avg * 0.05):
        reasons.append("wait/sync")
    if op_type == "Convolution":
        if phase == "decode" or m == 1:
            reasons.append("decode-small-M")
        if weight_bits == "8":
            reasons.append("non-w4-weight")
        if weight_bits == "4" and "W4" not in kernel:
            reasons.append("w4-missed")
        if total_bw >= 20 and gflops < 120:
            reasons.append("memory-bound")
        elif total_bw < 10 and gflops < 120:
            reasons.append("low-utilization")
    elif op_type == "Raster":
        reasons.append("layout-copy")
    elif op_type in ("BinaryOp", "UnaryOp", "LayerNorm", "Reduction", "Cast"):
        reasons.append("elementwise/memory")
    elif op_type == "Attention":
        reasons.append("attention")
    return ",".join(reasons) if reasons else "compute"


def print_table(title, rows, headers, limit=None):
    if limit is not None:
        rows = rows[:limit]
    print("\n" + title)
    if not rows:
        print("(none)")
        return
    widths = [len(h) for h in headers]
    formatted = []
    for row in rows:
        values = [str(row.get(h, "")) for h in headers]
        formatted.append(values)
        widths = [max(w, len(v)) for w, v in zip(widths, values)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for values in formatted:
        print("  ".join(v.ljust(w) for v, w in zip(values, widths)))


def main():
    parser = argparse.ArgumentParser(description="Analyze MNN op profile CSV and point out likely bottlenecks.")
    parser.add_argument("csv", nargs="?", default="logs/qwen3_vl_img_op_profile_phase.csv")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--min-ms", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.csv, newline="") as fp:
        rows = list(csv.DictReader(fp))

    total = sum(f(r, "total_ms", f(r, "avg_ms")) for r in rows)
    print(f"CSV: {args.csv}")
    print(f"Rows: {len(rows)}")
    print(f"Profiled op time: {total:.3f} ms")

    phase_stats = defaultdict(lambda: defaultdict(float))
    phase_type = defaultdict(lambda: defaultdict(float))
    kernel_stats = defaultdict(lambda: defaultdict(float))
    type_stats = defaultdict(lambda: defaultdict(float))
    issue_stats = defaultdict(lambda: defaultdict(float))

    for r in rows:
        phase = s(r, "phase", "unknown")
        op_type = s(r, "op_type", "unknown")
        kernel = s(r, "kernel", "-") or "-"
        issue = classify(r)
        add_stat(phase_stats, phase, r)
        add_stat(phase_type, (phase, op_type), r)
        add_stat(kernel_stats, (op_type, kernel), r)
        add_stat(type_stats, op_type, r)
        for part in issue.split(","):
            add_stat(issue_stats, part, r)

    phase_rows = []
    for key, v in phase_stats.items():
        phase_rows.append({
            "phase": key,
            "ms": f"{v['time']:.2f}",
            "%": f"{pct(v['time'], total):.2f}",
            "act_bw": f"{gbps(v['io'], v['time']):.2f}",
            "total_bw": f"{gbps(v['io'] + v['static'], v['time']):.2f}",
        })
    phase_rows.sort(key=lambda x: float(x["ms"]), reverse=True)
    print_table("Phase Summary", phase_rows, ["phase", "ms", "%", "act_bw", "total_bw"])

    type_rows = []
    for key, v in type_stats.items():
        type_rows.append({
            "op_type": key,
            "ms": f"{v['time']:.2f}",
            "%": f"{pct(v['time'], total):.2f}",
            "calls": int(v["count"]),
            "act_bw": f"{gbps(v['io'], v['time']):.2f}",
            "total_bw": f"{gbps(v['io'] + v['static'], v['time']):.2f}",
        })
    type_rows.sort(key=lambda x: float(x["ms"]), reverse=True)
    print_table("Op Type Summary", type_rows, ["op_type", "ms", "%", "calls", "act_bw", "total_bw"])

    issue_rows = []
    for key, v in issue_stats.items():
        issue_rows.append({
            "bottleneck_hint": key,
            "ms": f"{v['time']:.2f}",
            "%": f"{pct(v['time'], total):.2f}",
            "calls": int(v["count"]),
        })
    issue_rows.sort(key=lambda x: float(x["ms"]), reverse=True)
    print_table("Likely Bottleneck Classes", issue_rows, ["bottleneck_hint", "ms", "%", "calls"])

    top_rows = []
    hot = [r for r in rows if f(r, "total_ms", f(r, "avg_ms")) >= args.min_ms]
    hot.sort(key=lambda r: f(r, "total_ms", f(r, "avg_ms")), reverse=True)
    for r in hot[:args.top]:
        mkn = ""
        if s(r, "gemm_m"):
            mkn = f"{s(r, 'gemm_m')}x{s(r, 'gemm_k')}x{s(r, 'gemm_n')}"
        top_rows.append({
            "total_ms": f"{f(r, 'total_ms', f(r, 'avg_ms')):.3f}",
            "avg_ms": f"{f(r, 'avg_ms'):.3f}",
            "%": f"{pct(f(r, 'total_ms', f(r, 'avg_ms')), total):.2f}",
            "phase": s(r, "phase", "unknown"),
            "op": s(r, "op_type"),
            "node": s(r, "node_name")[-64:],
            "MKN": mkn,
            "kernel": s(r, "kernel"),
            "bw": f"{f(r, 'total_bw_gbps'):.2f}",
            "gflops": f"{f(r, 'gflops'):.1f}",
            "hint": classify(r),
        })
    print_table("Top Hot Ops", top_rows, ["total_ms", "avg_ms", "%", "phase", "op", "node", "MKN", "kernel", "bw", "gflops", "hint"])

    phase_type_rows = []
    for (phase, op_type), v in phase_type.items():
        phase_type_rows.append({
            "phase": phase,
            "op_type": op_type,
            "ms": f"{v['time']:.2f}",
            "%": f"{pct(v['time'], total):.2f}",
            "calls": int(v["count"]),
        })
    phase_type_rows.sort(key=lambda x: float(x["ms"]), reverse=True)
    print_table("Phase x Op Type", phase_type_rows, ["phase", "op_type", "ms", "%", "calls"], limit=args.top)

    print("\nRecommendations")
    conv_decode = sum(v["time"] for (phase, op), v in phase_type.items() if phase == "decode" and op == "Convolution")
    raster = type_stats["Raster"]["time"] if "Raster" in type_stats else 0.0
    lm_head = [r for r in rows if "/lm/lm_head/Linear" in s(r, "node_name")]
    if lm_head:
        lm = max(lm_head, key=lambda r: f(r, "avg_ms"))
        print(f"- lm_head: {f(lm, 'total_ms', f(lm, 'avg_ms')):.2f} ms, phase={s(lm,'phase')}, MKN={s(lm,'gemm_m')}x{s(lm,'gemm_k')}x{s(lm,'gemm_n')}, weight_bits={s(lm,'weight_bits')}, kernel={s(lm,'kernel')}.")
        if s(lm, "weight_bits") == "8":
            print("  Consider checking why lm_head is not W4; vocab projection is a major decode hotspot.")
    if conv_decode > total * 0.15:
        print("- Decode Convolution is a major cost. Focus on M=1 GEMM kernels, lm_head, and thread scheduling overhead.")
    if raster > total * 0.05:
        print("- Raster is non-trivial. Inspect top Raster nodes for avoidable reshape/transpose/layout materialization.")
    missed_w4 = [r for r in rows if s(r, "op_type") == "Convolution" and s(r, "weight_bits") == "4" and "W4" not in s(r, "kernel")]
    if missed_w4:
        print(f"- {len(missed_w4)} W4-weight Convolution rows did not use a W4 kernel.")
    repacked = [r for r in rows if s(r, "repack_via_int8") == "true"]
    if repacked:
        print(f"- {len(repacked)} rows report repack_via_int8=true; check shape alignment for direct int4 reorder.")
    print("- Validate bandwidth with hardware counters if possible; CSV bandwidth is an effective estimate from tensor/weight bytes.")


if __name__ == "__main__":
    main()
