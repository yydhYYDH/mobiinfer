#!/usr/bin/env python3
"""Prepare replay-style multi-turn GUI-agent prompts from MobiMind JSONL."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Iterable


TASK_RE = re.compile(r"### Current Task\s*\n(?P<task>.+?)\n### Action History", re.S)
HISTORY_RE = re.compile(
    r"### Action History\s*\nThe sequence of actions you have already taken:\s*\n"
    r"(?P<history>.*?)\n\s*<image>",
    re.S,
)


@dataclass
class Step:
    system: str
    user: str
    assistant: str
    image: str
    task: str
    history_count: int


def canonical_json_text(text: str) -> str:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except json.JSONDecodeError:
        return re.sub(r"\s+", " ", text).strip()


def message_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, dict):
                chunks.append(str(item.get("text") or item.get("content") or ""))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return str(value)


def extract_messages(obj: dict) -> tuple[str, str, str]:
    system = user = assistant = ""
    for msg in obj.get("messages", []):
        role = msg.get("role")
        content = message_text(msg.get("content", ""))
        if role == "system" and not system:
            system = content
        elif role == "user" and not user:
            user = content
        elif role == "assistant":
            assistant = content
    return system, user, assistant


def extract_task(user: str) -> str:
    match = TASK_RE.search(user)
    if not match:
        return ""
    task = match.group("task").strip()
    if len(task) >= 2 and task[0] == '"' and task[-1] == '"':
        task = task[1:-1]
    return task


def history_count(user: str) -> int:
    if not HISTORY_RE.search(user):
        return -1
    return len(extract_history_entries(user))


def extract_history_entries(user: str) -> list[str]:
    match = HISTORY_RE.search(user)
    if not match:
        return []
    history = match.group("history").strip()
    if history == "(No history)":
        return []

    entries: list[str] = []
    positions = list(re.finditer(r"(?m)^\s*\d+\.\s*", history))
    for idx, pos in enumerate(positions):
        start = pos.end()
        end = positions[idx + 1].start() if idx + 1 < len(positions) else len(history)
        entry = history[start:end].strip()
        if entry:
            entries.append(entry)
    return entries


def follows_previous(prev: Step, current: Step) -> bool:
    if current.history_count != prev.history_count + 1:
        return False
    entries = extract_history_entries(current.user)
    if len(entries) != current.history_count:
        return False
    return canonical_json_text(entries[-1]) == canonical_json_text(prev.assistant)


def iter_steps(jsonl: Path) -> Iterable[Step]:
    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            system, user, assistant = extract_messages(obj)
            images = obj.get("images") or []
            image = images[0] if images else ""
            task = extract_task(user)
            hcount = history_count(user)
            if system and user and assistant and image and task and hcount >= 0:
                yield Step(system, user, assistant, image, task, hcount)


def split_flat_prompt(prompt: str) -> tuple[str, str]:
    marker = "\n### Current Task"
    pos = prompt.find(marker)
    if pos < 0:
        return "", prompt
    return prompt[:pos].rstrip(), prompt[pos:].lstrip()


def iter_sharegpt_steps(json_path: Path) -> Iterable[Step]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {json_path}")
    for obj in data:
        convs = obj.get("conversations") or []
        human = assistant = ""
        for conv in convs:
            sender = conv.get("from")
            value = message_text(conv.get("value", ""))
            if sender in {"human", "user"} and not human:
                human = value
            elif sender in {"gpt", "assistant"}:
                assistant = value
        image = obj.get("image") or ""
        system, user = split_flat_prompt(human)
        task = extract_task(user)
        hcount = history_count(user)
        if system and user and assistant and image and task and hcount >= 0:
            yield Step(system, user, assistant, image, task, hcount)


def iter_input_steps(path: Path) -> Iterable[Step]:
    if path.suffix == ".jsonl":
        yield from iter_steps(path)
    else:
        yield from iter_sharegpt_steps(path)


def normalize_user_image(user: str, image_phone_path: str, hw: str) -> str:
    image_tag = f"<img>{image_phone_path}<hw>{hw}</hw></img>"
    return user.replace("<image>", image_tag)


def full_prompt(step: Step, image_phone_path: str, hw: str) -> str:
    return step.system.rstrip() + "\n\n" + normalize_user_image(step.user.strip(), image_phone_path, hw).strip() + "\n"


def build_trajectories(steps: Iterable[Step], num: int, min_steps: int, max_steps: int) -> list[list[Step]]:
    trajectories: list[list[Step]] = []
    current: list[Step] = []
    seen_in_current: set[tuple[int, str, str]] = set()

    def flush():
        nonlocal current, seen_in_current
        if len(current) >= min_steps:
            trajectories.append(current[:max_steps])
        current = []
        seen_in_current = set()

    for step in steps:
        key = (step.history_count, Path(step.image).name, step.assistant)
        if current and step.task == current[-1].task:
            if key in seen_in_current:
                continue
            if follows_previous(current[-1], step):
                current.append(step)
                seen_in_current.add(key)
                if len(current) >= max_steps:
                    flush()
                    if len(trajectories) >= num:
                        break
                continue
        flush()
        if len(trajectories) >= num:
            break
        if step.history_count == 0:
            current = [step]
            seen_in_current = {key}

    if len(trajectories) < num:
        flush()
    return trajectories[:num]


def build_trajectories_by_task(steps: Iterable[Step], num: int, min_steps: int, max_steps: int) -> list[list[Step]]:
    by_task: dict[str, dict[int, list[Step]]] = defaultdict(lambda: defaultdict(list))
    for step in steps:
        by_task[step.task][step.history_count].append(step)

    trajectories: list[list[Step]] = []
    for task, by_history in by_task.items():
        if 0 not in by_history:
            continue
        for start in by_history[0]:
            traj: list[Step] = [start]
            while len(traj) < max_steps:
                candidates = by_history.get(traj[-1].history_count + 1, [])
                next_step = next((candidate for candidate in candidates if follows_previous(traj[-1], candidate)), None)
                if next_step is None:
                    break
                traj.append(next_step)
            if len(traj) >= min_steps:
                trajectories.append(traj)
            if len(trajectories) >= num:
                break
        if len(trajectories) >= num:
            break
    return trajectories


def image_source(step: Step, source_dirs: list[str]) -> Path | None:
    original = Path(step.image)
    if original.exists():
        return original
    for source_dir in source_dirs:
        candidate = Path(source_dir) / original.name
        if candidate.exists():
            return candidate
    return None


def write_outputs(args, trajectories: list[list[Step]]) -> None:
    out = Path(args.output)
    prompts_dir = out / "prompts"
    images_dir = out / "images"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = [
        "trajectory_id\tstep_id\thistory_count\tprompt_path\timage_path\ttask\tassistant"
    ]
    image_count = 0
    for traj_id, traj in enumerate(trajectories):
        traj_dir = prompts_dir / f"trajectory_{traj_id:03d}"
        traj_dir.mkdir(parents=True, exist_ok=True)
        for step_id, step in enumerate(traj):
            image_name = Path(step.image).name
            phone_image = f"{args.phone_root.rstrip('/')}/images/{image_name}"
            prompt_path = traj_dir / f"step_{step_id:03d}.txt"
            prompt_path.write_text(full_prompt(step, phone_image, args.hw), encoding="utf-8")
            if args.copy_images:
                dst = images_dir / image_name
                if not dst.exists():
                    src = image_source(step, args.image_source_dir)
                    if src is None:
                        raise FileNotFoundError(step.image)
                    shutil.copy2(src, dst)
                    image_count += 1
            rel_prompt = prompt_path.relative_to(out).as_posix()
            rel_image = f"images/{image_name}"
            manifest_rows.append(
                "\t".join(
                    [
                        f"{traj_id:03d}",
                        f"{step_id:03d}",
                        str(step.history_count),
                        rel_prompt,
                        rel_image,
                        step.task.replace("\t", " "),
                        step.assistant.replace("\t", " ").replace("\n", " "),
                    ]
                )
            )

    (out / "manifest.tsv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    summary = {
        "num_trajectories": len(trajectories),
        "num_steps": sum(len(t) for t in trajectories),
        "min_steps": min((len(t) for t in trajectories), default=0),
        "max_steps": max((len(t) for t in trajectories), default=0),
        "avg_steps": (
            sum(len(t) for t in trajectories) / len(trajectories) if trajectories else 0
        ),
        "copied_images": image_count,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="/temp/csm/Dataset-train/mobimind_e2e_train.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--phone-root", required=True)
    parser.add_argument("--num-trajectories", type=int, default=20)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--hw", default="600,270")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument(
        "--group-by-task",
        action="store_true",
        help="Group shuffled samples by Current Task and rebuild consecutive history_count chains.",
    )
    parser.add_argument(
        "--available-images",
        help="Optional newline-delimited list of available image basenames.",
    )
    parser.add_argument(
        "--image-source-dir",
        action="append",
        default=[],
        help="Optional directory used to resolve image basenames when original JSONL paths are missing.",
    )
    args = parser.parse_args()

    available_images = None
    if args.available_images:
        available_images = {
            line.strip()
            for line in Path(args.available_images).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    needs_filtering = args.copy_images or available_images is not None
    candidate_num = args.num_trajectories * 20 if needs_filtering else args.num_trajectories
    input_steps = iter_input_steps(Path(args.jsonl))
    builder = build_trajectories_by_task if args.group_by_task else build_trajectories
    trajectories = builder(input_steps, num=candidate_num, min_steps=args.min_steps, max_steps=args.max_steps)
    if needs_filtering:
        trajectories = [
            traj for traj in trajectories
            if all(
                (
                    Path(step.image).name in available_images
                    if available_images is not None
                    else image_source(step, args.image_source_dir) is not None
                )
                for step in traj
            )
        ][: args.num_trajectories]
    write_outputs(args, trajectories)
    print(json.dumps({
        "num_trajectories": len(trajectories),
        "num_steps": sum(len(t) for t in trajectories),
        "output": args.output,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
