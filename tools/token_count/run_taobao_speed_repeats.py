#!/usr/bin/env python3
import json
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path("/home/ma-user/workspace/csm/mobiinfer")
PROMPT = "/home/ma-user/workspace/csm/mobi-autoround/data/prompt_template/longde-no-reasoning/taobao.md"
LLM_DEMO = str(ROOT / "build/llm_demo")
OUT_DIR = ROOT / "token_count/taobao_speed_repeats_0625"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS = 5

MODELS = {
    "pruned_no_draft": "/temp/csm/mnn_export_model/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40_vocab_pruned_exp-w4g32-a8-vis-w8a8-ns256-MNN/config.json",
    "full_eagle_specforge_hqq": "/temp/csm/mnn_export_model/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40-w4g32-a8-vis-w8a8-ns256-MNN-lastest-main-eagle-specforge-eaglehqq/config.json",
    "pruned_eagle_supplied_hqq": "/temp/csm/mnn_export_model/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40_vocab_pruned_exp-w4g32-a8-vis-w8a8-ns256-MNN-eagle-pruned-specforge-eaglehqq/config.json",
    "pruned_eagle_aligned_target_only_hqq": "/temp/csm/mnn_export_model/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40_vocab_pruned_exp-w4g32-a8-vis-w8a8-ns256-MNN-eagle-aligned-target-only-eaglehqq/config.json",
}

patterns = {
    "prompt_tokens": re.compile(r"prompt tokens num = ([0-9.]+)"),
    "decode_tokens": re.compile(r"decode tokens num = ([0-9.]+)"),
    "prefill_time": re.compile(r"prefill time = ([0-9.]+) s"),
    "decode_time": re.compile(r" decode time = ([0-9.]+) s"),
    "prefill_speed": re.compile(r"prefill speed = ([0-9.]+) tok/s"),
    "decode_speed": re.compile(r" decode speed = ([0-9.]+) tok/s"),
}

def parse(text):
    row = {}
    for k, pat in patterns.items():
        m = pat.search(text)
        row[k] = float(m.group(1)) if m else None
    first_json = ""
    for line in text.splitlines():
        if line.startswith("{"):
            first_json = line[:300]
            break
    row["output_prefix"] = first_json
    return row

def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None

def stdev(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0

summary = {}
for name, cfg in MODELS.items():
    print(f"===== {name}", flush=True)
    rows = []
    for i in range(1, RUNS + 1):
        log_path = OUT_DIR / f"{name}_run{i}.log"
        cmd = [LLM_DEMO, cfg, PROMPT]
        with log_path.open("w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        row = parse(text)
        row["returncode"] = proc.returncode
        row["log"] = str(log_path)
        rows.append(row)
        ps = row.get(prefill_speed)
        ds = row.get(decode_speed)
        dt = row.get(decode_tokens)
        print(frun={i} rc={proc.returncode} prefill={ps} decode={ds} decode_tokens={dt}, flush=True)
    summary[name] = {
        "config": cfg,
        "runs": rows,
        "mean_prefill_speed": mean([r["prefill_speed"] for r in rows]),
        "std_prefill_speed": stdev([r["prefill_speed"] for r in rows]),
        "mean_decode_speed": mean([r["decode_speed"] for r in rows]),
        "std_decode_speed": stdev([r["decode_speed"] for r in rows]),
        "mean_prefill_time": mean([r["prefill_time"] for r in rows]),
        "mean_decode_time": mean([r["decode_time"] for r in rows]),
        "mean_decode_tokens": mean([r["decode_tokens"] for r in rows]),
        "output_prefixes": [r["output_prefix"] for r in rows],
    }

json_path = OUT_DIR / "summary.json"
json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
md_path = OUT_DIR / "summary.md"
lines = ["# Taobao Speed Repeats", "", f"Runs per model: {RUNS}", "", "| Model | Prefill speed mean +- std | Decode speed mean +- std | Mean decode tokens | Output prefix |", "| --- | ---: | ---: | ---: | --- |"]
for name, s in summary.items():
    prefixes = [p for p in s["output_prefixes"] if p]
    prefix = prefixes[0].replace("|", "\\|") if prefixes else ""
    lines.append(
        f"| {name} | {s[mean_prefill_speed]:.2f} +- {s[std_prefill_speed]:.2f} | "
        f"{s[mean_decode_speed]:.2f} +- {s[std_decode_speed]:.2f} | "
        f"{s[mean_decode_tokens]:.1f} | `{prefix}` |"
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("===== summary")
print(md_path.read_text(encoding="utf-8"))
print("json", json_path)
