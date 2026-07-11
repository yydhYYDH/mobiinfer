#!/usr/bin/env python3
"""
compare_debug_outputs.py

详细对比真机 OM vs MNN-CPU 输出 (从手机 <modelDir>/debug_outputs/ 拷贝出来),
做精度比较 + 数值/误差作图.

输入: <debug_dir>/ (runOmVsMnnRealCalibTest 在真机保存的产物)
    chunk{i}_out{j}_{label}_mnn_cpu.bin   raw float32
    chunk{i}_out{j}_{label}_mnn_npu.bin   raw float32 (可能无)
    chunk{i}_out{j}_{label}_om.bin         raw float32 (可能无)
    chunk{i}_meta.json                     每个 output 的 shape/numel/label

输出: <out_dir>/
    report.json               每 chunk×output 的 cos/rel_mse/absmax/max_abs_err
    chunk{i}_out{j}_{label}.png   4 子图: mnn_cpu 曲线 / om 曲线 / 逐点散点 / 误差曲线
    summary.png               6 chunk 的 cos 柱状图

用法:
    python3 compare_debug_outputs.py \\
        --debug_dir /temp/fdh/model_omc/debug_outputs \\
        --out_dir /temp/fdh/input_calib/debug_eval \\
        --ref mnn_cpu --tgt om
"""
import argparse
import glob
import json
import os
import re

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


_META_RE = re.compile(r"chunk(\d+)_meta\.json$")


def _load_meta(debug_dir):
    metas = {}
    for p in sorted(glob.glob(os.path.join(debug_dir, "chunk*_meta.json"))):
        m = _META_RE.search(p)
        if not m:
            continue
        ci = int(m.group(1))
        with open(p, "r", encoding="utf-8") as f:
            metas[ci] = json.load(f)
    return metas


def _load_bin(debug_dir, chunk_idx, out_idx, label, src):
    path = os.path.join(debug_dir, f"chunk{chunk_idx}_out{out_idx}_{label}_{src}.bin")
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=np.float32)


def _cos(a, b):
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def _metrics(ref, tgt):
    ref = ref.astype(np.float64)
    tgt = tgt.astype(np.float64)
    diff = ref - tgt
    abs_ref = np.abs(ref)
    max_ref = float(abs_ref.max()) if abs_ref.size else 0.0
    if max_ref < 1e-12:
        max_ref = 1e-12
    return {
        "cos": _cos(ref, tgt),
        "rel_mse": float(np.mean(diff ** 2) / (np.linalg.norm(ref) * np.linalg.norm(tgt) + 1e-12)) if ref.size else 0.0,
        "max_abs_err": float(np.abs(diff).max()) if diff.size else 0.0,
        "mean_abs_err": float(np.abs(diff).mean()) if diff.size else 0.0,
        "absmax_ref": max_ref,
        "absmax_tgt": float(np.abs(tgt).max()) if tgt.size else 0.0,
        "rel_max_err_pct": float(np.abs(diff).max() / max_ref * 100.0) if diff.size else 0.0,
        "n": int(ref.size),
    }


def _plot_output(out_path, ref, tgt, label, chunk_idx, out_idx, src_pair):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    n = min(len(ref), len(tgt))
    x = np.arange(n)
    ref_n = ref[:n]
    tgt_n = tgt[:n]
    diff = ref_n - tgt_n

    ax = axes[0, 0]
    ax.plot(x, ref_n, label=src_pair[0], linewidth=0.8, alpha=0.8)
    ax.plot(x, tgt_n, label=src_pair[1], linewidth=0.8, alpha=0.8)
    ax.set_title(f"chunk{chunk_idx} {label} (out{out_idx}): values")
    ax.set_xlabel("element idx")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(ref_n, tgt_n, s=2, alpha=0.4)
    lim = max(float(np.abs(ref_n).max()), float(np.abs(tgt_n).max()), 1e-6)
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1)
    ax.set_title(f"scatter {src_pair[1]} vs {src_pair[0]}")
    ax.set_xlabel(src_pair[0])
    ax.set_ylabel(src_pair[1])
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, diff, linewidth=0.8, color="tab:red")
    ax.set_title("per-element error (ref - tgt)")
    ax.set_xlabel("element idx")
    ax.set_ylabel("error")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    abs_diff = np.abs(diff)
    if abs_diff.size:
        ax.hist(abs_diff, bins=80, color="tab:purple", alpha=0.8)
        ax.set_title("abs error distribution")
    else:
        ax.set_title("empty")
    ax.set_xlabel("|ref - tgt|")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"chunk{chunk_idx} {label} (out{out_idx}): {src_pair[0]} vs {src_pair[1]}  "
                 f"cos={_cos(ref_n, tgt_n):.6f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug_dir", required=True, help="真机保存的 debug_outputs 目录")
    ap.add_argument("--out_dir", required=True, help="对比结果输出目录")
    ap.add_argument("--ref", default="mnn_cpu", help="参考源: mnn_cpu / mnn_npu")
    ap.add_argument("--tgt", default="om", help="对比源: om / mnn_npu")
    ap.add_argument("--chunks", default="", help="只比这些 chunk, 逗号分隔 (默认全部)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metas = _load_meta(args.debug_dir)
    if not metas:
        raise FileNotFoundError(f"no chunk*_meta.json under {args.debug_dir}")

    chunk_ids = sorted(metas.keys())
    if args.chunks.strip():
        wanted = [int(x) for x in args.chunks.split(",") if x.strip()]
        chunk_ids = [c for c in chunk_ids if c in wanted]

    print(f"ref={args.ref}  tgt={args.tgt}")
    print(f"chunks: {chunk_ids}")
    print(f"debug_dir: {args.debug_dir}")
    print(f"out_dir: {args.out_dir}")
    print("=" * 60)

    report = {"ref": args.ref, "tgt": args.tgt, "per_chunk": {}}
    summary_rows = []

    for ci in chunk_ids:
        meta = metas[ci]
        chunk_name = meta.get("chunk_name", "")
        outputs = meta.get("outputs", [])
        print(f"\n=== chunk {ci} ({chunk_name})  {len(outputs)} outputs ===")
        chunk_report = {"chunk_name": chunk_name, "outputs": {}}

        for out_spec in outputs:
            oi = out_spec["out"]
            label = out_spec["label"]
            shape_str = out_spec.get("shape", "")
            numel = out_spec.get("numel", 0)

            ref = _load_bin(args.debug_dir, ci, oi, label, args.ref)
            tgt = _load_bin(args.debug_dir, ci, oi, label, args.tgt)
            if ref is None:
                print(f"  out{oi} {label}: SKIP (no {args.ref} bin)")
                chunk_report["outputs"][str(oi)] = {"status": "no_ref"}
                continue
            if tgt is None:
                print(f"  out{oi} {label}: SKIP (no {args.tgt} bin)")
                chunk_report["outputs"][str(oi)] = {"status": "no_tgt"}
                continue

            # reshape per meta shape if possible
            shape = []
            if shape_str:
                shape = [int(d) for d in shape_str.split("x") if d]
            prod = 1
            for d in shape:
                prod *= d
            if shape and prod == ref.size:
                ref = ref.reshape(shape)
                tgt = tgt.reshape(shape)

            m = _metrics(ref.ravel(), tgt.ravel())
            chunk_report["outputs"][str(oi)] = {
                "label": label, "shape": shape_str, "numel": int(ref.size),
                **m,
            }
            summary_rows.append((ci, oi, label, m["cos"], m["rel_max_err_pct"]))
            print(f"  out{oi} {label:14s} shape={shape_str or '?':14s} "
                  f"cos={m['cos']:.6f} relMaxErr={m['rel_max_err_pct']:.4f}% "
                  f"absmax_ref={m['absmax_ref']:.3f} absmax_tgt={m['absmax_tgt']:.3f} "
                  f"maxAbsErr={m['max_abs_err']:.4e}")

            plot_path = os.path.join(args.out_dir,
                                     f"chunk{ci}_out{oi}_{label}.png")
            _plot_output(plot_path, ref.ravel(), tgt.ravel(), label, ci, oi,
                         (args.ref, args.tgt))

        report["per_chunk"][str(ci)] = chunk_report

    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    print("\n" + "=" * 60)
    print(f"report: {report_path}")

    # summary bar chart: cos per (chunk,output)
    if HAS_MPL and summary_rows:
        fig, ax = plt.subplots(figsize=(max(8, len(summary_rows) * 0.6), 5))
        labels = [f"c{c}o{o}\n{lab[:8]}" for c, o, lab, _, _ in summary_rows]
        cos_vals = [r[3] for r in summary_rows]
        colors = ["tab:green" if v > 0.999 else ("tab:orange" if v > 0.95 else "tab:red") for v in cos_vals]
        ax.bar(labels, cos_vals, color=colors)
        ax.set_ylabel("cosine similarity")
        ax.set_title(f"{args.ref} vs {args.tgt}: per output cosine")
        ax.set_ylim(min(0.9, min(cos_vals) - 0.01) if cos_vals else 0, 1.0005)
        ax.axhline(0.999, color="tab:green", linestyle="--", linewidth=0.8, label="0.999")
        ax.axhline(0.95, color="tab:orange", linestyle="--", linewidth=0.8, label="0.95")
        ax.legend()
        plt.xticks(rotation=45, ha="right", fontsize=8)
        fig.tight_layout()
        sp = os.path.join(args.out_dir, "summary.png")
        fig.savefig(sp, dpi=110)
        plt.close(fig)
        print(f"summary plot: {sp}")

    # console summary table
    print(f"\n{'chunk':>5} {'out':>4} {'label':>14} {'cos':>10} {'relMaxErr%':>12}")
    for c, o, lab, cos, rme in summary_rows:
        print(f"{c:>5} {o:>4} {lab:>14} {cos:>10.6f} {rme:>12.4f}")


if __name__ == "__main__":
    main()
