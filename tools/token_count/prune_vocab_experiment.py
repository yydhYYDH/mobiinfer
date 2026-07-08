#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer


DEFAULT_MODEL = "/temp/zhangdelong/models/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40"
DEFAULT_COUNTS = "/home/ma-user/workspace/csm/mobiinfer/token_count/jsonl_token_count_outputs/token_counts.csv"
DEFAULT_OUT = "/temp/zhangdelong/models/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40_vocab_pruned_exp"


def load_used_ids(path):
    ids = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.add(int(row["token_id"]))
    return ids


def copy_metadata(src, dst):
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name == "model.safetensors":
            continue
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isfile(s):
            shutil.copy2(s, d)


def build_keep(src, used_ids):
    with open(os.path.join(src, "tokenizer.json"), encoding="utf-8") as f:
        tokj = json.load(f)
    base_vocab = tokj["model"]["vocab"]
    id_to_token = {v: k for k, v in base_vocab.items()}
    base_vocab_size = len(base_vocab)
    added = tokj.get("added_tokens", [])
    added_ids = {int(x["id"]) for x in added}

    keep_base = {tid for tid in used_ids if tid < base_vocab_size}
    # Extra formatting tokens observed in prompt_template/longde-no-reasoning/taobao.md.
    # They have count 0 in mobimind_e2e_train.jsonl but preserve original newline segmentation.
    keep_base.update([515, 1837, 15752, 47989])
    # Keep every byte/base token. In this tokenizer these are the first 256 ids.
    keep_base.update(range(min(256, base_vocab_size)))

    merges = tokj["model"].get("merges", [])
    merge_map = {}
    for m in merges:
        if isinstance(m, str):
            parts = m.split()
        else:
            parts = m
        if len(parts) != 2:
            continue
        a, b = parts
        merged = a + b
        if merged in base_vocab:
            merge_map[merged] = (a, b)

    changed = True
    while changed:
        changed = False
        for tid in list(keep_base):
            token = id_to_token.get(tid)
            if token is None:
                continue
            pair = merge_map.get(token)
            if not pair:
                continue
            for part in pair:
                pid = base_vocab.get(part)
                if pid is not None and pid not in keep_base:
                    keep_base.add(pid)
                    changed = True

    keep_added = sorted(added_ids)
    keep_old_ids = sorted(keep_base) + keep_added
    old_to_new = {old: new for new, old in enumerate(keep_old_ids)}
    return tokj, keep_old_ids, old_to_new, base_vocab_size, added


def rewrite_tokenizer(src, dst, tokj, keep_old_ids, old_to_new, base_vocab_size, added):
    keep_base_ids = [i for i in keep_old_ids if i < base_vocab_size]
    old_base_vocab = tokj["model"]["vocab"]
    old_id_to_token = {v: k for k, v in old_base_vocab.items()}
    new_base_vocab = OrderedDict((old_id_to_token[old], old_to_new[old]) for old in keep_base_ids)
    tokj["model"]["vocab"] = dict(new_base_vocab)

    keep_tokens = set(new_base_vocab)
    new_merges = []
    for m in tokj["model"].get("merges", []):
        parts = m.split() if isinstance(m, str) else m
        if len(parts) != 2:
            continue
        a, b = parts
        if a in keep_tokens and b in keep_tokens and (a + b) in keep_tokens:
            new_merges.append(m)
    tokj["model"]["merges"] = new_merges

    new_added = []
    for item in added:
        old = int(item["id"])
        if old in old_to_new:
            new_item = dict(item)
            new_item["id"] = old_to_new[old]
            new_added.append(new_item)
    tokj["added_tokens"] = new_added
    with open(os.path.join(dst, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokj, f, ensure_ascii=False)

    added_map = {x["content"]: int(x["id"]) for x in new_added}
    with open(os.path.join(dst, "added_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(added_map, f, ensure_ascii=False, indent=2, sort_keys=True)

    # Update tokenizer_config added_tokens_decoder ids.
    tc_path = os.path.join(dst, "tokenizer_config.json")
    with open(tc_path, encoding="utf-8") as f:
        tc = json.load(f)
    decoder = {}
    for item in new_added:
        decoder[str(item["id"])] = {k: item[k] for k in ["content", "lstrip", "normalized", "rstrip", "single_word", "special"] if k in item}
    tc["added_tokens_decoder"] = decoder
    if "chat_template" in tc:
        pass
    with open(tc_path, "w", encoding="utf-8") as f:
        json.dump(tc, f, ensure_ascii=False, indent=2)


def remap_config(dst, old_to_new, new_vocab_size):
    path = os.path.join(dst, "config.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ["eos_token_id", "pad_token_id", "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"]:
        if key in cfg and cfg[key] in old_to_new:
            cfg[key] = old_to_new[cfg[key]]
    if "text_config" in cfg:
        cfg["text_config"]["vocab_size"] = new_vocab_size
        for key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            if key in cfg["text_config"] and cfg["text_config"][key] in old_to_new:
                cfg["text_config"][key] = old_to_new[cfg["text_config"][key]]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    gen_path = os.path.join(dst, "generation_config.json")
    if os.path.exists(gen_path):
        with open(gen_path, encoding="utf-8") as f:
            gen = json.load(f)
        for key in ["eos_token_id", "pad_token_id", "bos_token_id"]:
            if key not in gen:
                continue
            value = gen[key]
            if isinstance(value, list):
                gen[key] = [old_to_new[x] for x in value if x in old_to_new]
            elif value in old_to_new:
                gen[key] = old_to_new[value]
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(gen, f, ensure_ascii=False, indent=2)


def rewrite_weights(src, dst, keep_old_ids):
    model_path = os.path.join(src, "model.safetensors")
    out_path = os.path.join(dst, "model.safetensors")
    keep = torch.tensor(keep_old_ids, dtype=torch.long)
    tensors = {}
    metadata = None
    with safe_open(model_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            t = f.get_tensor(key)
            if key in ["lm_head.weight", "model.language_model.embed_tokens.weight"]:
                t = t.index_select(0, keep)
            tensors[key] = t
    save_file(tensors, out_path, metadata=metadata)


def validate(out, old_to_new, counts_path, limit=100):
    tok = AutoTokenizer.from_pretrained(out, trust_remote_code=True, local_files_only=True)
    print("new tokenizer len", len(tok))
    with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    print("new text vocab_size", cfg.get("text_config", {}).get("vocab_size"))
    old_top = []
    with open(counts_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            old_top.append((int(row["token_id"]), row["token"], int(row["count"])))
            if len(old_top) >= limit:
                break
    for old_id, _, _ in old_top[:10]:
        new_id = old_to_new[old_id]
        print(old_id, "->", new_id, repr(tok.decode([new_id], clean_up_tokenization_spaces=False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--counts", default=DEFAULT_COUNTS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    used_ids = load_used_ids(args.counts)
    tokj, keep_old_ids, old_to_new, base_vocab_size, added = build_keep(args.model, used_ids)
    print("used_ids", len(used_ids))
    print("base_vocab_size", base_vocab_size, "added", len(added))
    print("keep_old_ids", len(keep_old_ids), "keep_base", sum(1 for i in keep_old_ids if i < base_vocab_size))
    print("first_special_old_new", [(x["content"], x["id"], old_to_new.get(x["id"])) for x in added[:5]])
    if args.dry_run:
        return

    if os.path.exists(args.out):
        raise SystemExit(f"Output exists: {args.out}")
    copy_metadata(args.model, args.out)
    rewrite_tokenizer(args.model, args.out, tokj, keep_old_ids, old_to_new, base_vocab_size, added)
    remap_config(args.out, old_to_new, len(keep_old_ids))
    rewrite_weights(args.model, args.out, keep_old_ids)
    with open(os.path.join(args.out, "old_to_new_token_id.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in old_to_new.items()}, f, indent=2)
    with open(os.path.join(args.out, "kept_old_token_ids.txt"), "w", encoding="utf-8") as f:
        for old in keep_old_ids:
            f.write(f"{old}\n")
    validate(args.out, old_to_new, args.counts)


if __name__ == "__main__":
    main()
