#!/usr/bin/env python3
"""
unpack_calib_npz.py

把真实校准输入 npz 解压成裸 float32 bin + meta json, 供鸿蒙 app 的
runOmVsMnnRealCalibTest 在真机上读 (app 无 zip 解压能力, 不加新依赖).

每个 chunk 取指定 sample_idx 的 npz, 解出 3 个 key:
  hidden_states_in  [1, S, 1024]
  rotary_pos_emb    [2, S, 1, 64]
  attention_mask    [1, S, S]
存成裸 fp32 bin (文件名带 chunk 索引), 并写一份 calib_meta.json 记录 shape,
app 侧 fread 后按 shape 构造 VARP.

产物放 <out_dir>/ 下 (不分子目录), 供 push 到手机模型目录的 calib/ 子目录.

用法:
  python3 unpack_calib_npz.py \
      --npz_dir /temp/fdh/input_calib/calib_inputs_256 \
      --out_dir /temp/fdh/model_omc/calib \
      --sample_idx 0 --chunks 0,1,2,3,4,5
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="校准 npz 目录 (calib_inputs_256)")
    ap.add_argument("--out_dir", required=True, help="输出裸 bin + meta 的目录")
    ap.add_argument("--sample_idx", type=int, default=0, help="取每个 chunk 的第几个样本")
    ap.add_argument("--chunks", default="0,1,2,3,4,5", help="解哪些 chunk, 逗号分隔")
    args = ap.parse_args()

    chunk_ids = [int(x) for x in args.chunks.split(",") if x.strip() != ""]
    os.makedirs(args.out_dir, exist_ok=True)

    # meta 用扁平 key (app 侧的 extractJsonString/Int 只能读扁平 key,
    # 不支持嵌套). 形如: chunk0_hidden_shape="1,608,1024", chunk0_hidden_numel=...
    meta = {"sample_idx": args.sample_idx, "dtype": "f32"}
    total_bytes = 0
    for ci in chunk_ids:
        npz_name = f"chunk_{ci:02d}_sample_{args.sample_idx:03d}.npz"
        npz_path = os.path.join(args.npz_dir, npz_name)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"missing npz: {npz_path}")
        data = np.load(npz_path)
        # 转 fp32 (app 侧 chunk 按 fp32 跑, readVarToFloatVector 读 std::vector<float>)
        for key, bin_tag in [
            ("hidden_states_in", "hidden"),
            ("rotary_pos_emb", "rotary"),
            ("attention_mask", "mask"),
        ]:
            arr = data[key].astype(np.float32)
            bin_name = f"calib_chunk{ci}_{bin_tag}.bin"
            bin_path = os.path.join(args.out_dir, bin_name)
            arr.tofile(bin_path)
            shape_str = ",".join(str(int(d)) for d in arr.shape)
            meta[f"chunk{ci}_{bin_tag}_file"] = bin_name
            meta[f"chunk{ci}_{bin_tag}_shape"] = shape_str
            meta[f"chunk{ci}_{bin_tag}_numel"] = int(arr.size)
            total_bytes += arr.nbytes
        print(f"chunk{ci}: {npz_name} -> 3 bin, "
              f"hidden={meta[f'chunk{ci}_hidden_shape']} "
              f"rotary={meta[f'chunk{ci}_rotary_shape']} "
              f"mask={meta[f'chunk{ci}_mask_shape']}")

    meta_path = os.path.join(args.out_dir, "calib_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nmeta: {meta_path}")
    print(f"total: {len(chunk_ids)} chunks, {total_bytes / 1e6:.1f} MB (fp32)")


if __name__ == "__main__":
    main()
