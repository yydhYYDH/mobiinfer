#!/usr/bin/env python3
"""
bin_to_chunk_npz.py

把 MNN 引擎在 MNN_VISUAL_CHUNK_INPUT_DUMP 模式下 dump 出来的 per-chunk 校准输入
(.bin + meta .json) 转换成 plugin-quant 校准链路期望的 chunk_XX_sample_YYY.npz 格式。

输入目录结构(由 omni.cpp 的 dump 路径产生):
    <dump_dir>/
        hidden_states_in_chunk{i}_sample{j}.bin   raw float32
        rotary_pos_emb_chunk{i}_sample{j}.bin     raw float32
        attention_mask_chunk{i}_sample{j}.bin     raw float32
        meta_chunk{i}_sample{j}.json              {shape, dtype, file, ok}

输出目录结构(对齐 collect_visual_act_stats_v6.py 的约定,直接喂给
visual_plugin_quant_matmul_route.py --input_dir):
    <out_dir>/
        chunk_{i:02d}_sample_{j:03d}.npz
            hidden_states_in   [1, S, 1024]  fp16
            rotary_pos_emb     [2, S, 1, 64] fp16
            attention_mask     [1, S, S]     fp16
        visual_calib_manifest.json

要点:
- hidden_states_in: MNN 的 chunk0 输入是 [S, 1024](rank-2,来自 visual_pre 的
  patch_embed),chunk>=1 是 [1, S, 1024]。统一 reshape 成 [1, S, 1024] 以匹配
  npz 校准格式(chunk module forward 期望带 batch 维)。
- rotary_pos_emb: MNN 已经在 dump 时 squeeze(1),输出就是 [2, S, 1, 64],与
  OM/npz 格式一致,直接 reshape 即可。
- attention_mask: [1, S, S],直接用。
- dtype 统一转 fp16,与现有 calib_inputs 完全一致。
"""

import argparse
import glob
import json
import os
import re

import numpy as np


_META_RE = re.compile(r"meta_chunk(\d+)_sample(\d+)\.json$")


def _load_tensor(dump_dir, fname, shape):
    path = os.path.join(dump_dir, fname)
    arr = np.fromfile(path, dtype=np.float32)
    expected = 1
    for d in shape:
        expected *= d
    if arr.size != expected:
        raise ValueError(
            f"{fname}: float count {arr.size} != product(shape={shape})={expected}"
        )
    return arr.reshape(shape)


def _normalize_hidden(arr):
    # 统一到 [1, S, hidden]
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True, help="omni.cpp dump 目录")
    ap.add_argument("--out_dir", required=True, help="输出的 calib_inputs 目录")
    ap.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="输出 npz 的 dtype,默认 fp16(与现有 calib_inputs 一致)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_dtype = np.float16 if args.dtype == "fp16" else np.float32

    metas = sorted(glob.glob(os.path.join(args.dump_dir, "meta_chunk*_sample*.json")))
    if not metas:
        raise FileNotFoundError(f"no meta_chunk*_sample*.json under {args.dump_dir}")

    manifest = []
    for meta_path in metas:
        m = _META_RE.search(meta_path)
        if not m:
            continue
        chunk_idx = int(m.group(1))
        sample_idx = int(m.group(2))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        inputs = meta["inputs"]

        tensors = {}
        for key in ("hidden_states_in", "rotary_pos_emb", "attention_mask"):
            spec = inputs.get(key, {})
            if not spec.get("ok", False):
                raise RuntimeError(f"chunk{chunk_idx} sample{sample_idx} {key} dump failed")
            arr = _load_tensor(args.dump_dir, spec["file"], spec["shape"])
            if key == "hidden_states_in":
                arr = _normalize_hidden(arr)
            tensors[key] = arr.astype(out_dtype)

        out_name = f"chunk_{chunk_idx:02d}_sample_{sample_idx:03d}.npz"
        out_path = os.path.join(args.out_dir, out_name)
        np.savez(
            out_path,
            hidden_states_in=tensors["hidden_states_in"],
            rotary_pos_emb=tensors["rotary_pos_emb"],
            attention_mask=tensors["attention_mask"],
        )
        manifest.append(
            {
                "chunk": chunk_idx,
                "sample_idx": sample_idx,
                "file": out_path,
                "source": "real_mnn_dump",
                "shapes": {k: list(v.shape) for k, v in tensors.items()},
                "dtypes": {k: str(v.dtype) for k, v in tensors.items()},
                "stats": {
                    "hidden_absmax": float(np.abs(tensors["hidden_states_in"]).max()),
                    "rotary_min": float(tensors["rotary_pos_emb"].min()),
                    "rotary_max": float(tensors["rotary_pos_emb"].max()),
                },
            }
        )
        print(f"  chunk{chunk_idx} sample{sample_idx} -> {out_name}  "
              f"hidden_absmax={manifest[-1]['stats']['hidden_absmax']:.4f}")

    manifest_path = os.path.join(args.out_dir, "visual_calib_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nmanifest: {manifest_path}")
    print(f"total: {len(manifest)} samples across chunks "
          f"{sorted({m['chunk'] for m in manifest})}")


if __name__ == "__main__":
    main()
