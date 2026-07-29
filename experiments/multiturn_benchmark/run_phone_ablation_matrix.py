#!/usr/bin/env python3
"""Run and summarize the on-device multi-turn ablation matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


VALID_ACTIONS = {
    "click",
    "swipe",
    "click_input",
    "input",
    "open_app",
    "press_home",
    "press_back",
    "wait",
    "done",
}


@dataclass(frozen=True)
class Variant:
    name: str
    config: str
    mode: str
    pruned: bool
    d3: bool
    kv: bool


VARIANTS = {
    item.name: item
    for item in [
        Variant("baseline", "bench20/configs/config_baseline_w8a8_ar.json", "raw-ar", False, False, False),
        Variant("d3_only", "bench20/configs/config_baseline_w8a8_la_d3_bin.json", "raw", False, True, False),
        Variant("kv_only", "bench20/configs/config_baseline_w8a8_ar.json", "cached-history-ar", False, False, True),
        Variant("prune_only", "bench20/configs/config_vocab_pruned_w8a8_ar.json", "raw-ar", True, False, False),
        Variant("prune_kv", "bench20/configs/config_vocab_pruned_w8a8_ar.json", "cached-history-ar", True, False, True),
        Variant("prune_d3", "bench20/configs/config_vocab_pruned_w8a8_la_d3_bin.json", "raw", True, True, False),
        Variant("full", "bench20/configs/config_vocab_pruned_w8a8_la_d3_bin.json", "cached-history", True, True, True),
    ]
}


STAT_PATTERNS = {
    "trajectories": (int, r"trajectories num = (\d+)"),
    "num_steps": (int, r"steps num = (\d+)"),
    "logical_prompt_tokens": (int, r"logical prompt tokens num = (\d+)"),
    "actual_prefill_tokens": (int, r"actual prefill tokens num = (\d+)"),
    "decode_tokens": (int, r"decode tokens num = (\d+)"),
    "tokenize_time_s": (float, r"tokenize time = ([0-9.]+) s"),
    "vision_time_s": (float, r"vision time = ([0-9.]+) s"),
    "pixels_mp": (float, r"pixels_mp = ([0-9.]+) MP"),
    "prefill_time_s": (float, r"prefill time = ([0-9.]+) s"),
    "decode_time_s": (float, r"decode time = ([0-9.]+) s"),
    "wall_generate_time_s": (float, r"wall generate time = ([0-9.]+) s"),
    "effective_prefill_speed": (float, r"effective prefill speed = ([0-9.]+) tok/s"),
    "actual_prefill_speed": (float, r"actual prefill speed = ([0-9.]+) tok/s"),
    "decode_speed": (float, r"decode speed = ([0-9.]+) tok/s"),
    "prefill_token_reduction": (float, r"prefill token reduction = ([0-9.]+)"),
    "draft_tokens": (int, r"lookahead draft tokens num = (\d+)"),
    "accepted_draft_tokens": (int, r"lookahead accepted draft tokens num = (\d+)"),
    "draft_accept_rate_pct": (float, r"lookahead draft accept rate = ([0-9.]+)%"),
    "full_accept_rate_pct": (float, r"lookahead full accept rate = ([0-9.]+)%"),
    "token_boundary_fallbacks": (int, r"token boundary fallbacks num = (\d+)"),
    "errors": (int, r"errors num = (\d+)"),
}


def parse_csv_list(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def serialize_manifest(rows: list[dict[str, str]]) -> str:
    if not rows:
        raise ValueError("Cannot serialize an empty manifest")
    fields = list(rows[0])
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(row[field].replace("\t", " ").replace("\n", " ") for field in fields))
    return "\n".join(lines) + "\n"


def choose_trajectories(
    rows: list[dict[str, str]], max_steps: int, tasks: int, requested: list[str]
) -> list[str]:
    counts = Counter(row["trajectory_id"] for row in rows)
    if requested:
        missing = [item for item in requested if counts[item] < max_steps]
        if missing:
            raise ValueError(f"Trajectories do not have {max_steps} steps: {missing}")
        selected = requested
    else:
        selected = sorted(
            (trajectory for trajectory, count in counts.items() if count >= max_steps),
            key=lambda trajectory: (-counts[trajectory], trajectory),
        )
    if len(selected) < tasks:
        raise ValueError(
            f"Requested {tasks} trajectories with at least {max_steps} steps, found {len(selected)}"
        )
    return selected[:tasks]


def ensure_subset(
    source_bench: Path,
    output_root: Path,
    all_rows: list[dict[str, str]],
    trajectories: list[str],
    step_length: int,
) -> tuple[Path, list[dict[str, str]]]:
    subset = output_root / "subsets" / f"steps_{step_length}"
    subset.mkdir(parents=True, exist_ok=True)
    selected = []
    for trajectory in trajectories:
        trajectory_rows = sorted(
            (row for row in all_rows if row["trajectory_id"] == trajectory),
            key=lambda row: int(row["step_id"]),
        )
        selected.extend(trajectory_rows[:step_length])
    expected_manifest = serialize_manifest(selected)
    manifest_path = subset / "manifest.tsv"
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != expected_manifest:
            raise RuntimeError(f"Existing subset manifest differs: {manifest_path}")
    else:
        manifest_path.write_text(expected_manifest, encoding="utf-8")

    for name in ("prompts", "images"):
        link = subset / name
        target = source_bench / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    return subset, selected


def extract_json_object(text: str) -> tuple[dict | None, bool]:
    stripped = text.strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("action") in VALID_ACTIONS:
            noisy = bool(stripped[:start].strip() or stripped[start + end :].strip())
            return obj, noisy
    return None, False


def verify_outputs(text: str, rows: list[dict[str, str]]) -> dict:
    pattern = re.compile(
        r"\[MODEL_OUTPUT_BEGIN\]\s+([^\n]+)\n(.*?)\n\[MODEL_OUTPUT_END\]\s+\1",
        re.DOTALL,
    )
    blocks = dict(pattern.findall(text))
    expected = {
        f"{row['trajectory_id']}/{row['step_id']}": json.loads(row["assistant"])["action"]
        for row in rows
    }
    valid = 0
    clean = 0
    matches = 0
    details = []
    for label, expected_action in expected.items():
        obj, noisy = extract_json_object(blocks.get(label, ""))
        action = obj.get("action") if obj else None
        if action is not None:
            valid += 1
            if not noisy:
                clean += 1
        if action == expected_action:
            matches += 1
        details.append(
            {"label": label, "expected": expected_action, "actual": action, "noisy": noisy}
        )
    total = len(expected)
    return {
        "output_blocks": len(blocks),
        "expected_blocks": total,
        "valid_json_actions": valid,
        "clean_json_actions": clean,
        "gold_action_matches": matches,
        "format_pass": valid == total and len(blocks) == total,
        "clean_json_pass": clean == total and len(blocks) == total,
        "gold_action_pass": matches == total,
        "format_rate": valid / total if total else 0.0,
        "clean_json_rate": clean / total if total else 0.0,
        "gold_action_match_rate": matches / total if total else 0.0,
        "action_details": details,
    }


def parse_stats(text: str) -> dict:
    parsed = {}
    for key, (cast, pattern) in STAT_PATTERNS.items():
        match = re.search(pattern, text)
        parsed[key] = cast(match.group(1)) if match else None
    if parsed["tokenize_time_s"] is not None and parsed["wall_generate_time_s"] is not None:
        parsed["e2e_time_s"] = parsed["tokenize_time_s"] + parsed["wall_generate_time_s"]
    else:
        parsed["e2e_time_s"] = None
    return parsed


def read_thermal_c() -> dict:
    values = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            value = float(path.read_text().strip())
        except (OSError, ValueError):
            continue
        if value > 1000:
            value /= 1000.0
        if 0 < value < 150:
            values.append(value)
    battery = None
    battery_path = Path("/sys/class/power_supply/battery/temp")
    try:
        battery = float(battery_path.read_text().strip())
        if battery > 100:
            battery /= 10.0
    except (OSError, ValueError):
        pass
    return {
        "thermal_max_c": max(values) if values else None,
        "thermal_avg_c": sum(values) / len(values) if values else None,
        "battery_temp_c": battery,
    }


def next_log_path(log_dir: Path, stem: str, rerun: bool) -> Path | None:
    primary = log_dir / f"{stem}.log"
    if not primary.exists():
        return primary
    text = primary.read_text(errors="ignore")
    if not rerun and "#################################" in text:
        return None
    retry = 1
    while (log_dir / f"{stem}.retry{retry}.log").exists():
        retry += 1
    return log_dir / f"{stem}.retry{retry}.log"


def run_one(
    args: argparse.Namespace,
    variant: Variant,
    step_length: int,
    repeat: int,
    subset: Path,
    rows: list[dict[str, str]],
) -> dict | None:
    log_dir = args.output / "logs"
    result_dir = args.output / "run_results"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = f"r{repeat:02d}_s{step_length}_{variant.name}"
    log_path = next_log_path(log_dir, stem, args.rerun)
    result_path = result_dir / f"{stem}.json"
    if log_path is None and result_path.exists():
        print(f"skip complete: {stem}", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    command = [
        str(args.binary),
        variant.config,
        str(subset),
        variant.mode,
        str(args.max_new_tokens),
    ]
    print("run:", " ".join(command), flush=True)
    if args.dry_run:
        return None

    thermal_before = read_thermal_c()
    start = time.time()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(args.binary.parent)
    env["MOBIINFER_PRINT_OUTPUT"] = "1"
    return_code = -1
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                command,
                cwd=args.root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_seconds,
                check=False,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    elapsed = time.time() - start
    thermal_after = read_thermal_c()
    text = log_path.read_text(errors="replace")
    result = {
        "variant": variant.name,
        "pruned": variant.pruned,
        "d3": variant.d3,
        "kv": variant.kv,
        "mode": variant.mode,
        "config": variant.config,
        "step_length": step_length,
        "repeat": repeat,
        "trajectory_ids": sorted({row["trajectory_id"] for row in rows}),
        "log": str(log_path),
        "return_code": return_code,
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "thermal_before": thermal_before,
        "thermal_after": thermal_after,
        **parse_stats(text),
        **verify_outputs(text, rows),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"done: {stem} format={result['format_rate']:.3f} "
        f"gold={result['gold_action_match_rate']:.3f} e2e={result['e2e_time_s']}",
        flush=True,
    )
    return result


def flatten_result(result: dict) -> dict:
    flat = {key: value for key, value in result.items() if key not in {"action_details", "thermal_before", "thermal_after"}}
    flat["trajectory_ids"] = ",".join(result["trajectory_ids"])
    for prefix in ("thermal_before", "thermal_after"):
        for key, value in result[prefix].items():
            flat[f"{prefix}_{key}"] = value
    return flat


def summarize_results(output: Path) -> dict:
    results = []
    for path in sorted((output / "run_results").glob("*.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    if not results:
        return {"runs": 0, "groups": []}

    flat_rows = [flatten_result(result) for result in results]
    csv_path = output / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    grouped = defaultdict(list)
    for result in results:
        if result.get("errors") == 0 and not result.get("timed_out"):
            grouped[(result["variant"], result["step_length"])].append(result)

    metric_keys = [
        "logical_prompt_tokens",
        "actual_prefill_tokens",
        "decode_tokens",
        "tokenize_time_s",
        "vision_time_s",
        "prefill_time_s",
        "decode_time_s",
        "wall_generate_time_s",
        "e2e_time_s",
        "effective_prefill_speed",
        "actual_prefill_speed",
        "decode_speed",
        "prefill_token_reduction",
        "draft_accept_rate_pct",
        "full_accept_rate_pct",
        "format_rate",
        "clean_json_rate",
        "gold_action_match_rate",
    ]
    summaries = []
    by_name_steps = {}
    for (variant, step_length), items in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        summary = {"variant": variant, "step_length": step_length, "runs": len(items)}
        for key in metric_keys:
            values = [item[key] for item in items if item.get(key) is not None]
            summary[f"median_{key}"] = statistics.median(values) if values else None
        summaries.append(summary)
        by_name_steps[(variant, step_length)] = summary

    comparisons = []
    pairs = [
        ("d3_full_vocab", "baseline", "d3_only"),
        ("pruning", "baseline", "prune_only"),
        ("pruning_with_d3", "d3_only", "prune_d3"),
        ("kv_full_vocab", "baseline", "kv_only"),
        ("kv_pruned_ar", "prune_only", "prune_kv"),
        ("d3_no_kv", "prune_only", "prune_d3"),
        ("d3_with_kv", "prune_kv", "full"),
        ("kv_with_d3", "prune_d3", "full"),
        ("full_system", "baseline", "full"),
    ]
    step_lengths = sorted({step_length for _, step_length in by_name_steps})
    for step_length in step_lengths:
        for component, baseline, optimized in pairs:
            left = by_name_steps.get((baseline, step_length))
            right = by_name_steps.get((optimized, step_length))
            if not left or not right:
                continue
            comparison = {
                "component": component,
                "step_length": step_length,
                "baseline": baseline,
                "optimized": optimized,
            }
            for metric in ("prefill_time_s", "decode_time_s", "e2e_time_s"):
                left_value = left.get(f"median_{metric}")
                right_value = right.get(f"median_{metric}")
                comparison[f"{metric}_speedup"] = (
                    left_value / right_value if left_value and right_value else None
                )
            comparisons.append(comparison)

    summary = {"runs": len(results), "groups": summaries, "comparisons": comparisons}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def reparse_existing_results(
    output: Path, subsets: dict[int, tuple[Path, list[dict[str, str]]]]
) -> int:
    updated = 0
    for result_path in sorted((output / "run_results").glob("*.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        log_path = Path(result["log"])
        if not log_path.exists():
            continue
        step_length = int(result["step_length"])
        rows = subsets[step_length][1]
        text = log_path.read_text(errors="replace")
        result.update(parse_stats(text))
        result.update(verify_outputs(text, rows))
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        updated += 1
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/data/com.termux/files/home/csm/mobiinfer"))
    parser.add_argument("--source-bench", default="bench_multiturn_20traj_4step")
    parser.add_argument(
        "--binary",
        default="experiments/multiturn_benchmark/bin_426e9a43/multiturn_bench_demo",
    )
    parser.add_argument("--output", default="bench_multiturn_ablation_426e9a43")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--step-lengths", default="1,2,3,4,5")
    parser.add_argument("--trajectory-ids", default="")
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--reparse-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    args.root = args.root.resolve()
    args.source_bench = (args.root / args.source_bench).resolve()
    args.binary = (args.root / args.binary).resolve()
    args.output = (args.root / args.output).resolve()
    variant_names = parse_csv_list(args.variants)
    unknown = [name for name in variant_names if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    variants = [VARIANTS[name] for name in variant_names]
    step_lengths = sorted(set(parse_csv_list(args.step_lengths, int)))
    requested_trajectories = parse_csv_list(args.trajectory_ids)
    if not step_lengths or min(step_lengths) < 1:
        raise ValueError("Step lengths must be positive")
    if not args.binary.exists():
        raise FileNotFoundError(args.binary)

    all_rows = load_manifest(args.source_bench / "manifest.tsv")
    trajectories = choose_trajectories(
        all_rows, max(step_lengths), args.tasks, requested_trajectories
    )
    args.output.mkdir(parents=True, exist_ok=True)
    subsets = {
        step_length: ensure_subset(
            args.source_bench,
            args.output,
            all_rows,
            trajectories,
            step_length,
        )
        for step_length in step_lengths
    }
    plan = {
        "variants": variant_names,
        "step_lengths": step_lengths,
        "trajectory_ids": trajectories,
        "tasks": args.tasks,
        "repeats": args.repeats,
        "cooldown_seconds": args.cooldown_seconds,
        "seed": args.seed,
    }
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise RuntimeError(f"Existing plan differs: {plan_path}")
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)

    if args.reparse_existing:
        print(f"reparsed={reparse_existing_results(args.output, subsets)}", flush=True)
    if args.summarize_only:
        print(json.dumps(summarize_results(args.output), ensure_ascii=False, indent=2), flush=True)
        return

    for repeat in range(args.repeats):
        for step_length in step_lengths:
            order = variants.copy()
            random.Random(args.seed + repeat * 1000 + step_length).shuffle(order)
            subset, rows = subsets[step_length]
            for variant in order:
                run_one(args, variant, step_length, repeat, subset, rows)
                if not args.dry_run and args.cooldown_seconds > 0:
                    time.sleep(args.cooldown_seconds)
                summarize_results(args.output)
    summary = summarize_results(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
