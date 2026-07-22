#!/usr/bin/env python3
"""Verify multiturn_bench_demo debug outputs.

The runner prints blocks like:

  [MODEL_OUTPUT_BEGIN] trajectory/step
  ...
  [MODEL_OUTPUT_END] trajectory/step

This script fails if any block is empty, cannot be parsed as a JSON object, or
does not contain a known GUI-agent action.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
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


def extract_json_object(text: str) -> tuple[dict | None, bool]:
    text = text.strip()
    if not text:
        return None, False
    for start in [idx for idx, ch in enumerate(text) if ch == "{"]:
        decoder = json.JSONDecoder()
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("action") in VALID_ACTIONS:
            has_prefix_or_suffix = bool(text[:start].strip() or text[start + end :].strip())
            return obj, has_prefix_or_suffix
    return None, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-steps", type=int, default=4)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    expected_actions: dict[str, str] = {}
    if args.manifest:
        with args.manifest.open(newline="", encoding="utf-8") as manifest_file:
            for row in csv.DictReader(manifest_file, delimiter="\t"):
                label = f"{row['trajectory_id']}/{row['step_id']}"
                expected_actions[label] = json.loads(row["assistant"])["action"]

    content = args.log.read_text(errors="replace")
    pattern = re.compile(
        r"\[MODEL_OUTPUT_BEGIN\]\s+([^\n]+)\n(.*?)\n\[MODEL_OUTPUT_END\]\s+\1",
        re.DOTALL,
    )
    blocks = pattern.findall(content)
    failures: list[str] = []
    warnings: list[str] = []

    if len(blocks) != args.expected_steps:
        failures.append(f"expected {args.expected_steps} output blocks, found {len(blocks)}")

    for label, text in blocks:
        obj, noisy = extract_json_object(text)
        if obj is None:
            preview = " ".join(text.strip().split())[:160]
            failures.append(f"{label}: output is not a valid JSON object; preview={preview!r}")
            continue
        action = obj.get("action")
        if action not in VALID_ACTIONS:
            failures.append(f"{label}: invalid action {action!r}")
        expected_action = expected_actions.get(label)
        if expected_action is not None and action != expected_action:
            failures.append(
                f"{label}: action mismatch: expected {expected_action!r}, got {action!r}"
            )
        if noisy:
            warnings.append(f"{label}: valid JSON action found, but output has extra prefix/suffix text")

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        for warning in warnings:
            print(f"- WARN: {warning}")
        return 1

    print("PASS" if not warnings else "PASS_WITH_WARNINGS")
    for label, text in blocks:
        obj, noisy = extract_json_object(text)
        print(f"- {label}: action={obj.get('action')}")
    for warning in warnings:
        print(f"- WARN: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
