#!/usr/bin/env python3
"""
collect_visual_act_stats_v6.py

为 visual chunk 量化实验收集激活统计：
1) 支持直接读取已有的 chunk 输入 .npz；
2) 如果未提供输入目录，则自动生成格式正确的随机校准输入；
3) 输出每个 visual block 线性层的输入 min/max/absmax 与默认 16bit 对称量化参数。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
EXPORT_DIR = os.path.dirname(SCRIPT_DIR)
HELPER_DIR = os.path.join(EXPORT_DIR, "pytorch_visual_omc6_route_v1")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, EXPORT_DIR)
sys.path.insert(0, HELPER_DIR)

# Ensure yaspin stub before any model-loading imports (mirrors _ensure_yaspin_stub in the main script)
import types as _types

try:
    import yaspin  # noqa: F401
except ImportError:
    _mod = _types.ModuleType("yaspin")

    class _DummyYaspin:
        def __init__(self, text="", color=None):
            self.text = text
            self.color = color

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def hide(self):
            return None

        def show(self):
            return None

        def fail(self, *_args, **_kwargs):
            return None

        def ok(self, *_args, **_kwargs):
            return None

    _mod.yaspin = _DummyYaspin
    sys.modules["yaspin"] = _mod

from export_visual_onnxv6 import (
    ExportAttentionAdapter,
    apply_visual_gptq_fake_quant,
    build_chunk_specs,
    generate_random_calibration_sample,
    make_chunk_module,
    make_dummy_args,
)


def _reset_kv(module):
    for sub in module.modules():
        if hasattr(sub, "past_key_value"):
            sub.past_key_value = None


def _to_tensor_dict(sample_npz):
    return {
        "hidden_states_in": torch.from_numpy(sample_npz["hidden_states_in"]),
        "rotary_pos_emb": torch.from_numpy(sample_npz["rotary_pos_emb"]),
        "attention_mask": torch.from_numpy(sample_npz["attention_mask"]),
    }


def _save_sample(sample, file_path):
    np.savez(
        file_path,
        hidden_states_in=sample["hidden_states_in"].cpu().numpy(),
        rotary_pos_emb=sample["rotary_pos_emb"].cpu().numpy(),
        attention_mask=sample["attention_mask"].cpu().numpy(),
    )


def _sample_file_name(chunk_idx, sample_idx):
    return f"chunk_{chunk_idx:02d}_sample_{sample_idx:03d}.npz"


def _get_chunk_sample(args, chunk_idx, sample_idx):
    if args.input_dir:
        file_path = os.path.join(args.input_dir, _sample_file_name(chunk_idx, sample_idx))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing calibration sample: {file_path}")
        return _to_tensor_dict(np.load(file_path)), file_path, "provided"

    seed = args.seed + chunk_idx * 1000 + sample_idx
    sample = generate_random_calibration_sample(
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        rotary_dim=args.rotary_dim,
        dtype=torch.float16 if args.dtype == "fp16" else torch.float32,
        seed=seed,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    file_path = os.path.join(args.output_dir, _sample_file_name(chunk_idx, sample_idx))
    _save_sample(sample, file_path)
    return sample, file_path, "random"


def _update_stats(stats, name, tensor):
    x = tensor.detach().to(torch.float32).cpu()
    cur = stats.setdefault(
        name,
        {
            "min": float("inf"),
            "max": float("-inf"),
            "absmax": 0.0,
            "num_samples": 0,
            "shape_examples": [],
        },
    )
    cur["min"] = min(cur["min"], float(x.min().item()))
    cur["max"] = max(cur["max"], float(x.max().item()))
    cur["absmax"] = max(cur["absmax"], float(x.abs().max().item()))
    cur["num_samples"] += 1
    shape = list(x.shape)
    if shape not in cur["shape_examples"] and len(cur["shape_examples"]) < 4:
        cur["shape_examples"].append(shape)


def _register_block_hooks(chunk_module, start_block_idx, stats):
    handles = []

    def add_hook(module, name):
        def _hook(_mod, inputs):
            if not inputs:
                return
            _update_stats(stats, name, inputs[0])

        handles.append(module.register_forward_pre_hook(_hook))

    for local_idx, block in enumerate(chunk_module.blocks):
        global_idx = start_block_idx + local_idx
        attn = getattr(block, "self_attn", None)
        if isinstance(attn, ExportAttentionAdapter):
            attn = attn.attn
        if attn is not None:
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                linear = getattr(attn, attr, None)
                if linear is not None:
                    add_hook(linear, f"blocks.{global_idx}.self_attn.{attr}")

        mlp = getattr(block, "mlp", None)
        if mlp is not None:
            for attr in ("linear_fc1", "linear_fc2"):
                linear = getattr(mlp, attr, None)
                if linear is not None:
                    add_hook(linear, f"blocks.{global_idx}.mlp.{attr}")

    return handles


def _finalize_stats(stats):
    finalized = {}
    for name, item in sorted(stats.items()):
        absmax = max(item["absmax"], 1e-8)
        finalized[name] = {
            "bit": 16,
            "unsigned_quant": False,
            "min": item["min"],
            "max": item["max"],
            "absmax": absmax,
            "scale": absmax / 32767.0,
            "offset": 0.0,
            "num_samples": item["num_samples"],
            "shape_examples": item["shape_examples"],
        }
    return finalized


def main():
    parser = argparse.ArgumentParser(description="Collect visual block activation stats for fake quant experiments")
    parser.add_argument("--path", required=True, help="HF model path")
    parser.add_argument("--dst_path", required=True, help="Output directory for stats and random samples")
    parser.add_argument("--npu_chunks", type=int, default=6)
    parser.add_argument("--visual_gptq_path", default=None, help="Optional visual GPTQ path to patch fake-quant weights")
    parser.add_argument("--input_dir", default=None, help="Directory of existing calibration npz files")
    parser.add_argument("--num_samples", type=int, default=4, help="Samples per chunk")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--rotary_dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    os.makedirs(args.dst_path, exist_ok=True)
    args.output_dir = os.path.join(args.dst_path, "calib_inputs")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.path} ...")
    from utils.model import LlmModel

    model = LlmModel.from_pretrained(
        args.path,
        args=make_dummy_args(args.path, args.dst_path, args.visual_gptq_path),
    )

    if args.visual_gptq_path:
        report_path = os.path.join(args.dst_path, "visual_fake_quant_report.json")
        print(f"Applying visual GPTQ fake-quant weights from {args.visual_gptq_path} ...")
        apply_visual_gptq_fake_quant(model, args.visual_gptq_path, report_path=report_path)
        print(f"  fake-quant report: {report_path}")

    target_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    stats = {}
    manifest = []
    specs = build_chunk_specs(len(model.visual.blocks), args.npu_chunks)

    for s, e, ci in specs:
        chunk_module, _local_ds = make_chunk_module(model.visual, s, e)
        chunk_module = chunk_module.to(dtype=target_dtype)
        chunk_module.eval()
        handles = _register_block_hooks(chunk_module, s, stats)
        try:
            for sample_idx in range(args.num_samples):
                sample, file_path, source = _get_chunk_sample(args, ci, sample_idx)
                sample = {k: v.to(dtype=target_dtype) for k, v in sample.items()}
                manifest.append(
                    {
                        "chunk": ci,
                        "blocks": [s, e - 1],
                        "sample_idx": sample_idx,
                        "file": file_path,
                        "source": source,
                        "shapes": {
                            k: list(v.shape) for k, v in sample.items()
                        },
                    }
                )
                with torch.no_grad():
                    _reset_kv(chunk_module)
                    chunk_module(
                        sample["hidden_states_in"],
                        sample["rotary_pos_emb"],
                        sample["attention_mask"],
                    )
        finally:
            for handle in handles:
                handle.remove()

    stats_path = os.path.join(args.dst_path, "visual_act_stats.json")
    manifest_path = os.path.join(args.dst_path, "visual_calib_manifest.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(_finalize_stats(stats), f, ensure_ascii=False, indent=2)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Activation stats saved to: {stats_path}")
    print(f"Calibration manifest saved to: {manifest_path}")
    print(f"Calibration samples directory: {args.output_dir}")


if __name__ == "__main__":
    main()
