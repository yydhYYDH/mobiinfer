#!/usr/bin/env python3
"""
select_images.py

从校准图片目录随机采样 N 张图，生成 llm_demo 用的 image_prompt.txt。

每行一个 prompt，格式：
    <img><hw>600,270</hw>/abs/path/to/image.jpg</img>

其中 <hw>H,W</hw> 会被 omni.cpp::multimodeProcess 解析为 mVisionHeight=H, mVisionWidth=W，
再经 qwen2VisionProcess round 到 32 对齐：
    600,270 -> H=608, W=256 -> grid_h=38, grid_w=16 -> seq_len = 608

所有样本同一 seq_len 是必须的：导出 ONNX 用 samples[0] 固定 shape、无 dynamic_axes，
因此 256 个校准样本必须共享同一 seq_len，否则后续样本 shape 不匹配会报错。

用法：
    python3 select_images.py \
        --src_dir /temp/csm/sft-0422-quant-500-half-size \
        --out_prompt /temp/fdh/input_calib/image_prompt.txt \
        --num 256 \
        --hw 600,270 \
        --seed 42
"""
import argparse
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="校准图片源目录")
    ap.add_argument("--out_prompt", required=True, help="输出的 image_prompt.txt 路径")
    ap.add_argument("--num", type=int, default=256, help="采样图片数")
    ap.add_argument("--hw", default="600,270",
                    help="强制缩放尺寸 H,W(逗号分隔)，默认 600,270 -> seq_len=608")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    args = ap.parse_args()

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = sorted([f for f in os.listdir(args.src_dir)
                    if f.lower().endswith(exts)])
    if not files:
        raise FileNotFoundError(f"no image under {args.src_dir}")

    if args.num > len(files):
        raise ValueError(f"only {len(files)} images, but --num={args.num}")

    rng = random.Random(args.seed)
    picked = rng.sample(files, args.num)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prompt)), exist_ok=True)
    with open(args.out_prompt, "w", encoding="utf-8") as f:
        for name in picked:
            abspath = os.path.abspath(os.path.join(args.src_dir, name))
            f.write(f"<img><hw>{args.hw}</hw>{abspath}</img>\n")

    print(f"picked {len(picked)} / {len(files)} images")
    print(f"prompt file: {args.out_prompt}")
    print(f"hw override: {args.hw}  (seq_len will be 608 for 600,270)")


if __name__ == "__main__":
    main()
