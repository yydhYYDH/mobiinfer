#!/usr/bin/env python3
"""
Check whether NPU outputs with the same shape are byte-identical within each chunk.
This detects a known NPU backend bug where onCopyBuffer's fallback path maps
all same-size outputs to the first NPU buffer slot.

Usage:
  python3 tools/check_npu_output_duplicates.py /path/to/debug_outputs/

Output:
  - Per-chunk table showing which NPU outputs are identical
  - Per-chunk table showing NPU vs CPU precision for each output
  - Summary of suspected buggy chunks
"""

import os
import sys
import struct
import glob
import re
import numpy as np
from collections import defaultdict


def read_bin(path):
    return np.fromfile(path, dtype=np.float32)


def read_meta(meta_path):
    entries = {}
    if not os.path.exists(meta_path):
        return entries
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            if key.startswith('out'):
                idx = int(key[3:])
                parts = val.split(' shape=')
                label = parts[0].strip()
                shape = parts[1].strip() if len(parts) > 1 else 'unknown'
                entries[idx] = (label, shape)
    return entries


def compare_npu_outputs(debug_dir):
    print("=" * 100)
    print("NPU Output Duplicate Detection")
    print("=" * 100)
    print()

    npu_files = sorted(glob.glob(os.path.join(debug_dir, "chunk*_out*_npu.bin")))
    if not npu_files:
        print("ERROR: no NPU output files found in %s" % debug_dir)
        return

    # Group files by chunk
    chunks = defaultdict(list)
    for f in npu_files:
        basename = os.path.basename(f)
        m = re.match(r'chunk(\d+)_out(\d+)_([\dx]+)_npu\.bin', basename)
        if m:
            chunks[int(m.group(1))].append((int(m.group(2)), m.group(3), f))

    bug_chunks = []

    for chunk_idx in sorted(chunks.keys()):
        entries = chunks[chunk_idx]
        entries.sort()

        meta = read_meta(os.path.join(debug_dir, "chunk%d_meta.txt" % chunk_idx))

        # Group entries by shape
        by_shape = defaultdict(list)
        for out_idx, shape, f in entries:
            by_shape[shape].append((out_idx, f))

        # Load data
        loaded = {}
        for out_idx, shape, f in entries:
            loaded[out_idx] = read_bin(f)

        cpu_loaded = {}
        for out_idx, shape, f in entries:
            cpu_path = f.replace('_npu.bin', '_cpu.bin')
            if os.path.exists(cpu_path):
                cpu_loaded[out_idx] = read_bin(cpu_path)

        print("--- Chunk %d ---" % chunk_idx)
        if chunk_idx in meta:
            print("  Model: (see meta)")

        # Section 1: NPU-to-NPU comparison within same shape
        print("  [NPU duplicate check]")
        any_suspicious = False
        for shape, items in by_shape.items():
            if len(items) < 2:
                continue
            print("    shape=%s (%d outputs):" % (shape, len(items)))
            for i in range(len(items)):
                oi = items[i][0]
                label_i = meta.get(oi, ("out%d" % oi, ""))[0] if oi in meta else "out%d" % oi
                for j in range(i + 1, len(items)):
                    oj = items[j][0]
                    label_j = meta.get(oj, ("out%d" % oj, ""))[0] if oj in meta else "out%d" % oj

                    npu_identical = np.array_equal(loaded[oi], loaded[oj])
                    npu_max_diff = np.max(np.abs(loaded[oi].astype(np.float64) -
                                                  loaded[oj].astype(np.float64)))

                    cpu_identical = False
                    cpu_max_diff = float('nan')
                    if oi in cpu_loaded and oj in cpu_loaded:
                        cpu_identical = np.array_equal(cpu_loaded[oi], cpu_loaded[oj])
                        cpu_max_diff = np.max(np.abs(cpu_loaded[oi].astype(np.float64) -
                                                      cpu_loaded[oj].astype(np.float64)))

                    if npu_identical and not cpu_identical:
                        status = "SUSPECT BUG"
                        any_suspicious = True
                    elif npu_identical and cpu_identical:
                        status = "same (OK, CPU also identical)"
                    else:
                        status = "different (OK)"

                    print("      out%d(%s) vs out%d(%s):"
                          " NPU_identical=%s NPU_maxDiff=%.6e"
                          " CPU_identical=%s CPU_maxDiff=%.6e"
                          " -> %s" % (
                              oi, label_i, oj, label_j,
                              npu_identical, npu_max_diff,
                              cpu_identical, cpu_max_diff,
                              status))

        if any_suspicious:
            bug_chunks.append(chunk_idx)

        # Section 2: NPU vs CPU precision
        print("  [NPU vs CPU precision]")
        header = "    %-25s %12s %12s %12s %12s %12s" % (
            "output", "shape", "max|ref|", "maxDiff", "relErr%", "rmsDiff")
        print(header)
        print("    " + "-" * (len(header) - 4))
        for out_idx, shape, f in entries:
            label = meta.get(out_idx, ("out%d" % out_idx, ""))[0] if out_idx in meta else "out%d" % out_idx
            if out_idx not in cpu_loaded:
                print("    %-25s %12s  (no CPU reference)" % (label, shape))
                continue
            cpu = cpu_loaded[out_idx]
            npu = loaded[out_idx]
            if cpu.shape != npu.shape:
                print("    %-25s %12s  SHAPE MISMATCH" % (label, shape))
                continue

            diff = np.abs(cpu.astype(np.float64) - npu.astype(np.float64))
            max_ref = max(float(np.max(np.abs(cpu))), 1e-6)
            max_diff = float(np.max(diff))
            rel_err = max_diff / max_ref * 100
            rms = float(np.sqrt(np.mean(diff ** 2)))
            n_elements = int(cpu.size)
            fail_mask = diff / max_ref > 0.02
            fail_count = int(np.sum(fail_mask))

            print("    %-25s %12s %12.4f %12.4f %10.4f%% %12.6f  (fail=%d/%d %.2f%%)" % (
                label[:25], shape, max_ref, max_diff, rel_err, rms,
                fail_count, n_elements, fail_count / n_elements * 100))

        # Section 3: Correlation matrix for same-shape NPU outputs
        if any(True for shape, items in by_shape.items() if len(items) >= 2):
            print("  [NPU same-shape correlation]")
            for shape, items in by_shape.items():
                if len(items) < 2:
                    continue
                indices = [it[0] for it in items]
                labels = [meta.get(i, ("out%d" % i, ""))[0] for i in indices]
                print("    shape=%s:" % shape)
                print("      " + " ".join("%10s" % l[:10] for l in labels))
                for i in range(len(indices)):
                    row = []
                    for j in range(len(indices)):
                        if i == j:
                            row.append(1.0)
                        else:
                            corr = np.corrcoef(loaded[indices[i]], loaded[indices[j]])[0, 1]
                            row.append(corr)
                    print("      %10s " % labels[i][:10] +
                          " ".join("%10.4f" % v for v in row))
        print()

    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    if bug_chunks:
        print("SUSPECT BUG in chunks: %s" % ", ".join("chunk%d" % c for c in bug_chunks))
        print()
        print("In these chunks, NPU outputs with identical shapes are byte-for-byte")
        print("identical while CPU outputs for the same pairs differ. This means")
        print("NPU onCopyBuffer is likely mapping all same-size outputs to the")
        print("SAME NPU buffer slot (the elementSize fallback always picks index 0).")
        print()
        print("Root cause (NPUBackend.cpp onCopyBuffer):")
        print("  1. mMNNOutTensors[i] == srcTensor pointer comparison fails")
        print("  2. Fallback: for(i...) if(elementSize matches) { matchIndex = i; break; }")
        print("  3. Always picks i=0 (first matching elementSize), so all same-size")
        print("     outputs read from mOutputTensors[0] -> hidden_states data")
    else:
        print("No suspicious NPU output duplicates detected.")

    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    compare_npu_outputs(sys.argv[1])


if __name__ == "__main__":
    main()
