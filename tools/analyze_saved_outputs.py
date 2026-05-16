#!/usr/bin/env python3
"""
Analyze CPU vs NPU output files saved by runQwen3VlChunkModelTest.

Usage:
  python3 tools/analyze_saved_outputs.py /path/to/debug_outputs/

The debug_outputs directory should contain:
  chunkN_outM_SHAPE_cpu.bin   (raw float32 binary)
  chunkN_outM_SHAPE_npu.bin
  chunkN_meta.txt             (label → file mapping)

Output:
  - Per-chunk comparison table (maxDiff, relErr, rmsDiff, fail%)
  - Per-element error histograms (saved as PNG if matplotlib available)
  - Scatter plots of CPU vs NPU values
"""

import os
import sys
import struct
import glob
import re
import numpy as np


def read_bin(path):
    """Read a raw float32 binary file."""
    return np.fromfile(path, dtype=np.float32)


def read_meta(meta_path):
    """Parse a chunk meta file, return list of (out_index, label, shape_str)."""
    entries = []
    if not os.path.exists(meta_path):
        return entries
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            if key.startswith('out'):
                # out0=hidden_states shape=1x608x1024
                idx = int(key[3:])
                parts = val.split(' shape=')
                label = parts[0].strip()
                shape = parts[1].strip() if len(parts) > 1 else 'unknown'
                entries.append((idx, label, shape))
    entries.sort()
    return entries


def compare_vectors(cpu, npu, label=""):
    """Return dict of comparison metrics."""
    if cpu.shape != npu.shape:
        return {"error": "shape mismatch cpu=%s npu=%s" % (str(cpu.shape), str(npu.shape))}

    diff = np.abs(cpu - npu)
    max_ref = max(np.max(np.abs(cpu)), 1e-6)
    max_diff = np.max(diff)
    rel_err = max_diff / max_ref
    rms = np.sqrt(np.mean(diff ** 2))

    # Per-element fail count (diff / maxRef > 2%)
    fail_mask = diff / max_ref > 0.02
    fail_count = int(np.sum(fail_mask))
    fail_idx = int(np.argmax(diff)) if fail_count > 0 else -1

    return {
        "label": label,
        "size": int(cpu.size),
        "max_ref": float(max_ref),
        "max_diff": float(max_diff),
        "rel_err_pct": float(rel_err * 100),
        "rms_diff": float(rms),
        "fail_count": fail_count,
        "fail_pct": float(fail_count / cpu.size * 100),
        "max_diff_idx": fail_idx,
    }


def analyze_directory(debug_dir):
    """Main analysis entry point."""
    print("=" * 90)
    print("Analyzing: %s" % debug_dir)
    print("=" * 90)

    # Group files by chunk
    bin_pattern = os.path.join(debug_dir, "chunk*_out*_cpu.bin")
    cpu_files = sorted(glob.glob(bin_pattern))

    if not cpu_files:
        print("ERROR: no CPU output files found in %s" % debug_dir)
        return

    # Parse chunk index from filename: chunk1_out0_1x608x1024_cpu.bin
    chunks = {}
    for cpu_path in cpu_files:
        basename = os.path.basename(cpu_path)
        m = re.match(r'chunk(\d+)_out(\d+)_([\dx]+)_cpu\.bin', basename)
        if not m:
            continue
        chunk_idx = int(m.group(1))
        out_idx = int(m.group(2))
        shape_str = m.group(3)
        npu_path = cpu_path.replace('_cpu.bin', '_npu.bin')

        if not os.path.exists(npu_path):
            print("WARNING: missing NPU counterpart for %s" % basename)
            continue

        if chunk_idx not in chunks:
            chunks[chunk_idx] = []
        chunks[chunk_idx].append((out_idx, cpu_path, npu_path, shape_str, basename))

    # Print per-chunk comparison table
    for chunk_idx in sorted(chunks.keys()):
        entries = chunks[chunk_idx]
        entries.sort()

        # Try to read meta
        meta_path = os.path.join(debug_dir, "chunk%d_meta.txt" % chunk_idx)
        meta = {e[0]: e[1] for e in read_meta(meta_path)} if os.path.exists(meta_path) else {}

        print()
        print("--- Chunk %d ---" % chunk_idx)
        header = "  %-30s %12s %12s %10s %10s %10s %8s %s" % (
            "label", "max|ref|", "maxDiff", "relErr%", "rmsDiff", "failCount", "fail%", "PASS?")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for out_idx, cpu_path, npu_path, shape_str, basename in entries:
            cpu = read_bin(cpu_path)
            npu = read_bin(npu_path)
            label = meta.get(out_idx, "out%d" % out_idx)

            result = compare_vectors(cpu, npu, label)
            if "error" in result:
                print("  %-30s ERROR: %s" % (label, result["error"]))
                continue

            passed = result["rel_err_pct"] < 2.0
            status = "PASS" if passed else ("WARN" if result["rel_err_pct"] < 5.0 else "FAIL")

            print("  %-30s %12.4f %12.4f %9.4f%% %10.6f %10d %7.2f%% %s" % (
                result["label"][:30],
                result["max_ref"],
                result["max_diff"],
                result["rel_err_pct"],
                result["rms_diff"],
                result["fail_count"],
                result["fail_pct"],
                status))


def plot_chunk(debug_dir, chunk_idx, save_dir=None):
    """Generate scatter plots and error histograms for a specific chunk."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    cpu_files = sorted(glob.glob(os.path.join(debug_dir, "chunk%d_out*_cpu.bin" % chunk_idx)))

    entries = []
    for cpu_path in cpu_files:
        basename = os.path.basename(cpu_path)
        m = re.match(r'chunk(\d+)_out(\d+)_([\dx]+)_cpu\.bin', basename)
        if not m:
            continue
        out_idx = int(m.group(2))
        npu_path = cpu_path.replace('_cpu.bin', '_npu.bin')
        if os.path.exists(npu_path):
            entries.append((out_idx, cpu_path, npu_path, basename))

    entries.sort()
    n_entries = len(entries)
    if n_entries == 0:
        print("No entries to plot for chunk %d" % chunk_idx)
        return

    fig, axes = plt.subplots(n_entries, 2, figsize=(14, 4 * n_entries))
    if n_entries == 1:
        axes = axes.reshape(1, 2)

    for row, (out_idx, cpu_path, npu_path, basename) in enumerate(entries):
        cpu = read_bin(cpu_path)
        npu = read_bin(npu_path)
        result = compare_vectors(cpu, npu)

        # Left: CPU vs NPU scatter
        ax1 = axes[row, 0]
        # Subsample for scatter if too many points
        n_sample = min(len(cpu), 50000)
        idx_sample = np.random.choice(len(cpu), n_sample, replace=False)
        ax1.scatter(cpu[idx_sample], npu[idx_sample], s=1, alpha=0.3)
        rng = [min(np.min(cpu), np.min(npu)), max(np.max(cpu), np.max(npu))]
        ax1.plot(rng, rng, 'r--', linewidth=0.5)
        ax1.set_xlabel("CPU")
        ax1.set_ylabel("NPU")
        ax1.set_title("Chunk %d %s\nmaxDiff=%.4f relErr=%.2f%%" % (
            chunk_idx, result.get("label", "?"),
            result.get("max_diff", 0), result.get("rel_err_pct", 0)))

        # Right: error distribution histogram
        ax2 = axes[row, 1]
        diff = np.abs(cpu - npu)
        ax2.hist(diff, bins=100, edgecolor='none', alpha=0.7)
        ax2.axvline(result.get("rms_diff", 0), color='r', linestyle='--', label='RMS')
        ax2.set_xlabel("|CPU - NPU|")
        ax2.set_ylabel("count")
        ax2.set_title("Error distribution | n=%d  fail=%d (%.2f%%)" % (
            result.get("size", 0), result.get("fail_count", 0),
            result.get("fail_pct", 0)))
        ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(save_dir or debug_dir, "chunk%d_analysis.png" % chunk_idx)
    plt.savefig(out_path, dpi=120)
    print("Plot saved to: %s" % out_path)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    debug_dir = sys.argv[1]

    analyze_directory(debug_dir)

    # Optionally generate plots
    if "--plot" in sys.argv:
        for chunk_idx in [1, 2, 3, 4, 5, 6]:
            plot_chunk(debug_dir, chunk_idx, debug_dir)


if __name__ == "__main__":
    main()
