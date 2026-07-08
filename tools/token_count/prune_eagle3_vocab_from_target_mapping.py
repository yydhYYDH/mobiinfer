#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


CONFIG_TOKEN_KEYS = {"bos_token_id", "eos_token_id", "pad_token_id"}


def copy_metadata(src, dst):
    os.makedirs(dst, exist_ok=False)
    for name in os.listdir(src):
        if name == "model.safetensors":
            continue
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isfile(s):
            shutil.copy2(s, d)
        elif os.path.isdir(s) and name not in {".cache", "__pycache__"}:
            shutil.copytree(s, d)


def load_old_to_new(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data.items()}


def load_target_vocab_size(path, old_to_new):
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        if "text_config" in cfg and "vocab_size" in cfg["text_config"]:
            return int(cfg["text_config"]["vocab_size"])
        if "vocab_size" in cfg:
            return int(cfg["vocab_size"])
    return max(old_to_new.values()) + 1


def add_token_value(value, out):
    if isinstance(value, int):
        out.add(value)
    elif isinstance(value, list):
        for item in value:
            add_token_value(item, out)


def collect_config_old_token_ids(*paths):
    ids = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        for key in CONFIG_TOKEN_KEYS:
            if key in cfg:
                add_token_value(cfg[key], ids)
        if "text_config" in cfg and isinstance(cfg["text_config"], dict):
            for key in CONFIG_TOKEN_KEYS:
                if key in cfg["text_config"]:
                    add_token_value(cfg["text_config"][key], ids)
    return ids


def load_keep_target_ids(path):
    ids = set()
    if not path:
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(int(line.split()[0]))
    return ids


def load_count_limited_old_target_ids(counts_path, old_to_new, min_count=0, top_k=0):
    if not counts_path:
        return None
    rows = []
    with open(counts_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "token_id" not in reader.fieldnames or "count" not in reader.fieldnames:
            raise RuntimeError(f"{counts_path} must have token_id and count columns")
        for row in reader:
            old_id = int(row["token_id"])
            count = int(row["count"])
            if old_id not in old_to_new:
                continue
            if count < min_count:
                continue
            rows.append((old_id, count))
    rows.sort(key=lambda item: (-item[1], item[0]))
    if top_k and top_k > 0:
        rows = rows[:top_k]
    return {old_id for old_id, _ in rows}


def target_token_from_draft(draft_idx, d2t):
    return int(draft_idx + int(d2t[draft_idx]))


def remap_optional_token_id(cfg, key, old_to_new):
    if key not in cfg:
        return
    value = cfg[key]
    if isinstance(value, int) and value in old_to_new:
        cfg[key] = old_to_new[value]
    elif isinstance(value, list):
        cfg[key] = [old_to_new[x] for x in value if isinstance(x, int) and x in old_to_new]


def build_allowed_old_target_ids(args, old_to_new):
    allowed = load_count_limited_old_target_ids(
        args.target_counts,
        old_to_new,
        min_count=args.min_count,
        top_k=args.top_k_targets,
    )
    if allowed is None:
        allowed = set(old_to_new)
    allowed.update(load_keep_target_ids(args.keep_target_ids))
    allowed.update(collect_config_old_token_ids(args.target_config, os.path.join(args.eagle_model, "config.json")))
    allowed = {old_id for old_id in allowed if old_id in old_to_new}
    return allowed


def prune_eagle(eagle_model, out_eagle, old_to_new, new_target_vocab_size, allowed_old_target_ids):
    copy_metadata(eagle_model, out_eagle)

    state = load_file(os.path.join(eagle_model, "model.safetensors"), device="cpu")
    required = {"lm_head.weight", "d2t", "t2d"}
    missing = required.difference(state)
    if missing:
        raise RuntimeError(f"Eagle model is missing required tensors: {sorted(missing)}")

    old_d2t = state["d2t"].cpu()
    old_draft_vocab_size = int(old_d2t.numel())
    keep_old_draft_ids = []
    rows = []

    for old_draft_id in range(old_draft_vocab_size):
        old_target_id = target_token_from_draft(old_draft_id, old_d2t)
        if old_target_id not in old_to_new:
            continue
        if old_target_id not in allowed_old_target_ids:
            continue
        new_draft_id = len(keep_old_draft_ids)
        new_target_id = old_to_new[old_target_id]
        new_d2t_value = new_target_id - new_draft_id
        keep_old_draft_ids.append(old_draft_id)
        rows.append((old_draft_id, old_target_id, new_draft_id, new_target_id, new_d2t_value))

    if not keep_old_draft_ids:
        raise RuntimeError("No Eagle draft token remains after applying target mapping/count filter.")

    keep = torch.tensor(keep_old_draft_ids, dtype=torch.long)
    new_d2t = torch.tensor([row[4] for row in rows], dtype=torch.int64)
    new_t2d = torch.zeros((new_target_vocab_size,), dtype=torch.bool)
    for _, _, _, new_target_id, _ in rows:
        if not 0 <= new_target_id < new_target_vocab_size:
            raise RuntimeError(f"new_target_id out of range: {new_target_id} >= {new_target_vocab_size}")
        new_t2d[new_target_id] = True

    tensors = {}
    for key, tensor in state.items():
        if key == "lm_head.weight":
            tensors[key] = tensor.index_select(0, keep)
            print(f"prune {key}: {tuple(tensor.shape)} -> {tuple(tensors[key].shape)}")
        elif key == "d2t":
            tensors[key] = new_d2t
            print(f"rewrite d2t: {tuple(tensor.shape)} -> {tuple(new_d2t.shape)}")
        elif key == "t2d":
            tensors[key] = new_t2d
            print(f"rewrite t2d: {tuple(tensor.shape)} -> {tuple(new_t2d.shape)}")
        else:
            tensors[key] = tensor
    save_file(tensors, os.path.join(out_eagle, "model.safetensors"))

    cfg_path = os.path.join(out_eagle, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["vocab_size"] = new_target_vocab_size
    cfg["draft_vocab_size"] = len(keep_old_draft_ids)
    for key in CONFIG_TOKEN_KEYS:
        remap_optional_token_id(cfg, key, old_to_new)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_eagle, "eagle_token_mapping.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["old_draft_id", "old_target_id", "new_draft_id", "new_target_id", "new_d2t"])
        writer.writerows(rows)

    with open(os.path.join(out_eagle, "kept_old_draft_token_ids.txt"), "w", encoding="utf-8") as f:
        for old_draft_id in keep_old_draft_ids:
            f.write(f"{old_draft_id}\n")

    return len(keep_old_draft_ids)


def validate(out_eagle, new_target_vocab_size):
    state = load_file(os.path.join(out_eagle, "model.safetensors"), device="cpu")
    lm_head = state["lm_head.weight"]
    d2t = state["d2t"]
    t2d = state["t2d"]
    assert lm_head.shape[0] == d2t.shape[0], (lm_head.shape, d2t.shape)
    assert t2d.shape[0] == new_target_vocab_size, (t2d.shape, new_target_vocab_size)
    for draft_id in range(int(d2t.numel())):
        target_id = draft_id + int(d2t[draft_id])
        assert 0 <= target_id < new_target_vocab_size, (draft_id, target_id)
        assert bool(t2d[target_id]), (draft_id, target_id)
    print("validate ok")
    print("new draft vocab size:", int(d2t.numel()))
    print("new target vocab size:", new_target_vocab_size)


def main():
    parser = argparse.ArgumentParser(description="Prune Eagle3 draft vocab after target/base vocab has been pruned.")
    parser.add_argument("--eagle-model", required=True, help="Original Eagle3 model directory.")
    parser.add_argument("--target-old-to-new", required=True, help="old_to_new_token_id.json from pruned target model.")
    parser.add_argument("--target-config", help="config.json from pruned target model; used to read new vocab_size and keep special ids.")
    parser.add_argument("--new-target-vocab-size", type=int, help="Override new target vocab size.")
    parser.add_argument("--target-counts", help="Optional CSV with old target token_id,count. Further prunes Eagle to frequent target tokens.")
    parser.add_argument("--min-count", type=int, default=0, help="With --target-counts, keep target tokens with count >= this value.")
    parser.add_argument("--top-k-targets", type=int, default=0, help="With --target-counts, keep only top K counted target tokens after min-count.")
    parser.add_argument("--keep-target-ids", help="Optional text file of old target token ids to force keep in Eagle.")
    parser.add_argument("--out-eagle", required=True, help="Output pruned Eagle3 directory.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    old_to_new = load_old_to_new(args.target_old_to_new)
    new_target_vocab_size = args.new_target_vocab_size or load_target_vocab_size(args.target_config, old_to_new)
    allowed_old_target_ids = build_allowed_old_target_ids(args, old_to_new)

    print("old_to_new size:", len(old_to_new))
    print("new target vocab size:", new_target_vocab_size)
    print("allowed old target ids for Eagle:", len(allowed_old_target_ids))

    state = load_file(os.path.join(args.eagle_model, "model.safetensors"), device="cpu")
    d2t = state["d2t"].cpu()
    kept_after_target = 0
    kept_after_filter = 0
    for draft_id in range(int(d2t.numel())):
        target_id = target_token_from_draft(draft_id, d2t)
        if target_id in old_to_new:
            kept_after_target += 1
            if target_id in allowed_old_target_ids:
                kept_after_filter += 1
    print("old draft vocab size:", int(d2t.numel()))
    print("draft rows surviving target prune:", kept_after_target)
    print("draft rows surviving count filter:", kept_after_filter)
    if args.dry_run:
        return

    if os.path.exists(args.out_eagle):
        raise SystemExit(f"Output exists: {args.out_eagle}")
    prune_eagle(args.eagle_model, args.out_eagle, old_to_new, new_target_vocab_size, allowed_old_target_ids)
    validate(args.out_eagle, new_target_vocab_size)


if __name__ == "__main__":
    main()
