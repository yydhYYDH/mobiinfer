#!/usr/bin/env python3
import argparse
import os
import re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "/temp/zhangdelong/models/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40_vocab_pruned_exp"
DEFAULT_MD = "/home/ma-user/workspace/csm/mobi-autoround/data/prompt_template/longde-no-reasoning/taobao.md"
DEFAULT_IMAGE = "/home/ma-user/workspace/csm/mobi-autoround/data/taobao/taobao_1_4.jpg"


def split_rendered_md(md_text, image_path):
    # The md is already rendered with Qwen chat markers. Convert it back to two user messages
    # so Qwen3VLProcessor can insert the correct image tokens/pixel inputs.
    text = md_text.replace("<|im_start|>assistant", "").strip()
    parts = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, flags=re.S)
    if len(parts) < 2:
        raise ValueError(f"Expected at least 2 user turns in {DEFAULT_MD}, got {len(parts)}")

    first = parts[0].strip()
    second = parts[1].strip()
    second = re.sub(r"<img>.*?</img>", "", second, flags=re.S).strip()

    return [
        {"role": "user", "content": [{"type": "text", "text": first}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": second},
                {"type": "image", "image": image_path},
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Run a Taobao image prompt through the pruned Qwen3VL model.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--md", default=DEFAULT_MD)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not os.path.exists(args.md):
        raise FileNotFoundError(args.md)
    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)

    with open(args.md, encoding="utf-8") as f:
        md_text = f.read()

    messages = split_rendered_md(md_text, args.image)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(args.image).convert("RGB")
    inputs = processor(text=[rendered], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    print("model:", args.model)
    print("md:", args.md)
    print("image:", args.image, image.size)
    print("rendered prompt chars:", len(rendered))
    print("input_ids shape:", tuple(inputs["input_ids"].shape), "max_id:", int(inputs["input_ids"].max()))
    for key in ["pixel_values", "image_grid_thw"]:
        if key in inputs:
            print(f"{key} shape:", tuple(inputs[key].shape))

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_len:]
    output = processor.tokenizer.decode(new_tokens[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    print("generated ids:", new_tokens[0].tolist())
    print("output:")
    print(output)


if __name__ == "__main__":
    main()
