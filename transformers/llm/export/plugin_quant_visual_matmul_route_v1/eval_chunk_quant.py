#!/usr/bin/env python3
"""
eval_chunk_quant.py

Visual chunk 量化效果评估。

对照华为 CANN Kit 文档「量化效果评估」的插件方法:
对浮点 chunk 插入量化算子 (optimize_model) -> 加载校准后的 pth ->
set_quant_state + set_calibrate_state(False) -> 跑前向仿真。
区别于文档的「整模型 generate 看输出」, 这里是 chunk 子模块, 改成
**逐样本对比量化前后输出张量的余弦相似度 / MSE**, 更适合 chunk 级定量评估。

数据流:
  256 个真实输入 (calib_inputs_256)
    ├── fp chunk (浮点原 chunk)          -> out_fp   (hidden + deepstack_k)
    └── qmodel (optimize_model + calibrated pth + quant_state, calibrate off)
                                         -> out_quant (hidden + deepstack_k)
  对每个输出张量逐样本算 cos / MSE / absmax, 汇总成 eval_report.json

复用 visual_plugin_quant_matmul_route.py 的:
  load_visual_chunk / load_calibration_samples / reset_kv_cache /
  config_path / calibrated_state_path
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

# 复用同目录的主脚本里的函数
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from visual_plugin_quant_matmul_route import (  # noqa: E402
    build_chunk_specs,
    calibrated_state_path,
    config_path,
    load_calibration_samples,
    load_visual_chunk,
    reset_kv_cache,
)
from utils.model import LlmModel  # noqa: E402
from dopt.dopt_lm.do_opt import (  # noqa: E402
    optimize_model,
    set_calibrate_state,
    set_quant_state,
)


# ----------------------------- 指标 -----------------------------
def _flatten(t):
    return t.detach().to(torch.float32).reshape(-1)


def cosine(a, b):
    a = _flatten(a)
    b = _flatten(b)
    na = a.norm() + 1e-12
    nb = b.norm() + 1e-12
    return float((a * b).sum() / (na * nb))


def rel_mse(a, b):
    """归一化 MSE = mean((a-b)^2) / (|a|*|b| + eps). 越小越好, 0 = 完全一致."""
    a = _flatten(a)
    b = _flatten(b)
    denom = (a.norm() * b.norm()).item() + 1e-12
    return float(((a - b) ** 2).mean() / denom)


def absmax(t):
    return float(t.detach().abs().max())


def elem_err_stats(a, b):
    a = _flatten(a)
    b = _flatten(b)
    d = (a - b).abs()
    return {"max_abs_err": float(d.max()), "mean_abs_err": float(d.mean())}


# --------------------------- 模型构建 ---------------------------
# 关键: fp 基线和 quant 基线必须各自独立加载 HF 模型, 不能共享 visual.
# 原因: optimize_model(chunk, cfg) 会**原地**把 chunk 里的 nn.Linear 换成 QLinear,
# 若 fp chunk 和 quant chunk 共享同一份 visual.blocks, 优化后 fp chunk 也变成量化模型,
# 导致两者输出完全相同 (cos≈1.0, 无量化误差). 必须各加载一份独立的浮点模型.

def _specs_from_model(model_path, route_dir, npu_chunks):
    """加载一次模型, 返回 chunk 切分规格后就释放."""
    from visual_plugin_quant_matmul_route import make_dummy_args
    args = make_dummy_args(model_path, route_dir)
    model = LlmModel.from_pretrained(model_path, args=args)
    specs = build_chunk_specs(len(model.visual.blocks), npu_chunks)
    del model
    import gc; gc.collect()
    return specs


def build_fp_chunk(model_path, route_dir, npu_chunks, ci):
    """浮点基线 chunk: 独立加载一份 HF 模型, 原 Linear, 不做 optimize_model.
    返回 (chunk, meta). meta 是 load_visual_chunk 返回的字典, 含 local_deepstack_count."""
    chunk, meta = load_visual_chunk(model_path, route_dir, npu_chunks, ci)
    chunk.eval()
    return chunk, meta


def build_quant_chunk(model_path, route_dir, npu_chunks, ci):
    """量化 chunk: 独立加载一份 HF 模型 -> optimize_model 注入 quant_op
    -> 加载 calibrated pth -> 开 quant_state / 关 calibrate.
    返回 (qmodel, meta, load_info)."""
    chunk, meta = load_visual_chunk(model_path, route_dir, npu_chunks, ci)
    chunk.eval()

    cfg_path = config_path(route_dir, ci)
    qmodel = optimize_model(chunk, cfg_path)
    qmodel.eval()

    state_path = calibrated_state_path(route_dir, ci)
    state = torch.load(state_path, map_location="cpu")
    try:
        qmodel.load_state_dict(state, strict=True)
        load_info = {"strict": True, "missing": [], "unexpected": []}
    except RuntimeError as e:
        # dopt 版本差异可能导致个别 key 不匹配, 降级 strict=False
        missing, unexpected = qmodel.load_state_dict(state, strict=False)
        load_info = {"strict": False, "missing": sorted(list(missing)),
                     "unexpected": sorted(list(unexpected))}
        print(f"  [warn] strict=True 失败, 降级 strict=False: {e}", file=sys.stderr)

    set_quant_state(qmodel, weight_state=True, input_state=True)
    set_calibrate_state(qmodel, False)   # 关校准态, 用固化 scale 做 fake quant
    qmodel.eval()
    return qmodel, meta, load_info


# --------------------------- 输出归一 ---------------------------
def _as_list(out):
    """chunk forward 输出可能是单 tensor 或 tuple. 统一成 list[Tensor]."""
    if isinstance(out, (tuple, list)):
        return list(out)
    return [out]


def _output_names(n_total):
    """n_total = 1(hidden) + local_ds_count. 第一个是 hidden, 其余是 deepstack_k."""
    names = ["hidden_states"]
    for k in range(n_total - 1):
        names.append(f"deepstack_{k}")
    return names


# --------------------------- 单样本评估 ---------------------------
def eval_one_sample(chunk_fp, qmodel, sample):
    """跑 fp / quant 两次前向, 返回 {output_name: {cos, rel_mse, absmax_fp, absmax_q, max_abs_err, mean_abs_err}}."""
    h = sample["hidden_states_in"]
    r = sample["rotary_pos_emb"]
    m = sample["attention_mask"]
    with torch.no_grad():
        reset_kv_cache(chunk_fp)
        out_fp = _as_list(chunk_fp(h, r, m))
        reset_kv_cache(qmodel)
        out_q = _as_list(qmodel(h, r, m))

    n = max(len(out_fp), len(out_q))
    result = {}
    for k in range(n):
        name = _output_names(n)[k]
        a = out_fp[k]
        b = out_q[k]
        st = elem_err_stats(a, b)
        result[name] = {
            "cos": cosine(a, b),
            "rel_mse": rel_mse(a, b),
            "absmax_fp": absmax(a),
            "absmax_q": absmax(b),
            "max_abs_err": st["max_abs_err"],
            "mean_abs_err": st["mean_abs_err"],
        }
    return result


# --------------------------- 汇总统计 ---------------------------
def _agg(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "n": int(arr.size),
    }


def aggregate(per_sample_metrics):
    """per_sample_metrics: list[dict[output_name -> dict]]. 返回 {output_name: {metric: agg}}."""
    if not per_sample_metrics:
        return {}
    out_names = list(per_sample_metrics[0].keys())
    metric_keys = list(per_sample_metrics[0][out_names[0]].keys())
    agg = {}
    for name in out_names:
        agg[name] = {}
        for mk in metric_keys:
            vals = [ps[name][mk] for ps in per_sample_metrics if name in ps and mk in ps[name]]
            agg[name][mk] = _agg(vals)
    return agg


# ------------------------------ main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/temp/models/mobi0402_2B_halfimage_rl",
                    help="HF 浮点模型路径 (与量化时一致)")
    ap.add_argument("--route_root", default="/temp/fdh/model_omc",
                    help="route_dir 的根目录")
    ap.add_argument("--route_suffix", default="real256",
                    help="route_dir 后缀: ${route_root}/model_visual_plugin_matmul_chunk{N}_${suffix}")
    ap.add_argument("--input_dir", default="/temp/fdh/input_calib/calib_inputs_256",
                    help="真实校准输入 npz 目录")
    ap.add_argument("--out_dir", default="/temp/fdh/input_calib/eval_real256",
                    help="评估产物输出目录")
    ap.add_argument("--npu_chunks", type=int, default=6)
    ap.add_argument("--chunks", default="0,1,2,3,4,5", help="评估哪些 chunk, 逗号分隔")
    ap.add_argument("--num_samples", type=int, default=256, help="每 chunk 评估几个样本 (0=全部)")
    args = ap.parse_args()

    chunk_ids = [int(x) for x in args.chunks.split(",") if x.strip() != ""]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"model_path : {args.model_path}")
    print(f"route_root : {args.route_root}  suffix={args.route_suffix}")
    print(f"input_dir  : {args.input_dir}")
    print(f"chunks     : {chunk_ids}  num_samples={args.num_samples}")
    print(f"out_dir    : {args.out_dir}")
    print("=" * 60)

    # 先拿 chunk 切分规格 (加载一次模型后释放; fp/quant 各自再独立加载, 避免原地污染)
    print("[load] 探测 chunk 规格 ...")
    specs = _specs_from_model(args.model_path, args.out_dir, args.npu_chunks)
    print(f"  specs = {[(s, e, ci) for s, e, ci in specs]}")

    summary = {"per_chunk": {}, "config": {
        "model_path": args.model_path,
        "input_dir": args.input_dir,
        "num_samples": args.num_samples,
        "chunks": chunk_ids,
    }}

    for ci in chunk_ids:
        spec = specs[ci]
        route_dir = os.path.join(args.route_root, f"model_visual_plugin_matmul_chunk{ci}_{args.route_suffix}")
        print(f"\n========== chunk {ci}  route={route_dir} ==========")

        # 前置检查: dopt_config + calibrated pth 必须存在
        cfg_path = config_path(route_dir, ci)
        pth_path = calibrated_state_path(route_dir, ci)
        missing_file = None
        for p in (cfg_path, pth_path):
            if not os.path.exists(p):
                missing_file = p
                break
        if missing_file is not None:
            print(f"  [skip] 缺失: {missing_file} (先跑 run_real_calib_256.sh)")
            summary["per_chunk"][str(ci)] = {"status": "missing", "missing_file": missing_file}
            continue

        # 构建 fp + quant (各自独立加载一份 HF 模型, 避免 optimize_model 原地污染)
        print(f"  [load] fp 基线 chunk ...")
        chunk_fp, meta_fp = build_fp_chunk(args.model_path, route_dir, args.npu_chunks, ci)
        print(f"  [load] 量化 qmodel (optimize_model + calibrated pth) ...")
        qmodel, meta_q, load_info = build_quant_chunk(args.model_path, route_dir, args.npu_chunks, ci)
        local_ds_count = meta_q["local_deepstack_count"]
        n_out = 1 + local_ds_count
        print(f"  block [{spec[0]},{spec[1]})  local_deepstack={local_ds_count}  outputs={_output_names(n_out)}")
        print(f"  load_info: strict={load_info['strict']} missing={len(load_info['missing'])} unexpected={len(load_info['unexpected'])}")

        # 加载样本
        samples, manifest = load_calibration_samples(
            args.input_dir, ci, args.num_samples, force_fp32=True
        )
        print(f"  samples: {len(samples)}")

        per_sample = []
        for idx, sample in enumerate(samples):
            m = eval_one_sample(chunk_fp, qmodel, sample)
            m["sample_idx"] = idx
            m["file"] = os.path.basename(manifest[idx]["file"])
            per_sample.append(m)
            if (idx + 1) % 32 == 0 or idx + 1 == len(samples):
                # 打印 hidden 的 cos 进度
                cos_h = m["hidden_states"]["cos"] if "hidden_states" in m else float("nan")
                print(f"    sample {idx + 1}/{len(samples)}  cos(hidden)={cos_h:.6f}")

        agg = aggregate([{k: v for k, v in ps.items() if k not in ("sample_idx", "file")}
                         for ps in per_sample])

        # 写 per-chunk 产物
        chunk_out = os.path.join(args.out_dir, f"chunk_{ci:02d}")
        os.makedirs(chunk_out, exist_ok=True)
        with open(os.path.join(chunk_out, "per_sample.csv"), "w", newline="") as f:
            out_names = [n for n in per_sample[0].keys() if n not in ("sample_idx", "file")]
            metric_keys = list(per_sample[0][out_names[0]].keys())
            header = ["sample_idx", "file"]
            for n in out_names:
                for mk in metric_keys:
                    header.append(f"{n}.{mk}")
            w = csv.writer(f)
            w.writerow(header)
            for ps in per_sample:
                row = [ps["sample_idx"], ps["file"]]
                for n in out_names:
                    for mk in metric_keys:
                        row.append(ps[n][mk])
                w.writerow(row)
        with open(os.path.join(chunk_out, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"aggregate": agg, "load_info": load_info,
                       "local_deepstack_count": local_ds_count,
                       "num_samples": len(samples)}, f, indent=2, ensure_ascii=False)

        # 控制台打印汇总
        print(f"  --- chunk {ci} 汇总 ---")
        for n in out_names:
            a = agg[n]
            print(f"    {n:14s}: cos mean={a['cos']['mean']:.6f} min={a['cos']['min']:.6f} "
                  f"rel_mse={a['rel_mse']['mean']:.2e} absmax_fp={a['absmax_fp']['mean']:.3f} "
                  f"absmax_q={a['absmax_q']['mean']:.3f}")

        summary["per_chunk"][str(ci)] = {
            "status": "ok",
            "block_start": spec[0],
            "block_end": spec[1],
            "local_deepstack_count": local_ds_count,
            "num_samples": len(samples),
            "load_info": load_info,
            "aggregate": agg,
        }

        # 释放显存/内存
        del chunk_fp, qmodel
        import gc
        gc.collect()

    # 写总报告
    report_path = os.path.join(args.out_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print("\n" + "=" * 60)
    print(f"eval_report: {report_path}")
    print("=" * 60)
    # 总览表
    print(f"{'chunk':>5} {'cos_hidden_mean':>16} {'cos_hidden_min':>15} {'status':>8}")
    for ci in chunk_ids:
        info = summary["per_chunk"].get(str(ci), {})
        if info.get("status") != "ok":
            print(f"{ci:>5} {'-':>16} {'-':>15} {info.get('status','?'):>8}")
            continue
        h = info["aggregate"].get("hidden_states", {}).get("cos", {})
        print(f"{ci:>5} {h.get('mean',float('nan')):>16.6f} {h.get('min',float('nan')):>15.6f} {'ok':>8}")


if __name__ == "__main__":
    main()
