#!/usr/bin/env python3
import argparse
import gc
import re
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessor

ORIG_MODEL = '/temp/zhangdelong/models/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40'
PRUNED_MODEL = '/temp/zhangdelong/models/MAI-UI-2B-0422-instruct-halfpixel-1ep_RLv2_4NPUS_bs128_ds5050_step40_vocab_pruned_exp'
DEFAULT_MD = '/home/ma-user/workspace/csm/mobi-autoround/data/prompt_template/longde-no-reasoning/taobao.md'
DEFAULT_IMAGE = '/home/ma-user/workspace/csm/mobi-autoround/data/taobao/taobao_1_4.jpg'


def split_rendered_md(md_text, image_path):
    text = md_text.replace('<|im_start|>assistant', '').strip()
    parts = re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', text, flags=re.S)
    if len(parts) < 2:
        raise ValueError(f'Expected at least 2 user turns, got {len(parts)}')
    first = parts[0].strip()
    second = re.sub(r'<img>.*?</img>', '', parts[1], flags=re.S).strip()
    return [
        {'role': 'user', 'content': [{'type': 'text', 'text': first}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': second}, {'type': 'image', 'image': image_path}]},
    ]


class StepTimer(LogitsProcessor):
    def __init__(self):
        self.times = []

    def __call__(self, input_ids, scores):
        self.times.append(time.perf_counter())
        return scores


def run_one(name, model_path, messages, image, max_new_tokens, dtype):
    print(f'\n=== {name} ===', flush=True)
    load_t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=dtype,
        device_map='cpu',
    )
    model.eval()
    load_s = time.perf_counter() - load_t0

    prep_t0 = time.perf_counter()
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors='pt')
    prep_s = time.perf_counter() - prep_t0

    print('model_path:', model_path)
    print('tokenizer_len:', len(processor.tokenizer))
    print('load_s:', round(load_s, 4))
    print('prepare_s:', round(prep_s, 4))
    print('input_ids_shape:', tuple(inputs['input_ids'].shape), 'max_id:', int(inputs['input_ids'].max()))
    if 'pixel_values' in inputs:
        print('pixel_values_shape:', tuple(inputs['pixel_values'].shape))
    if 'image_grid_thw' in inputs:
        print('image_grid_thw_shape:', tuple(inputs['image_grid_thw'].shape))

    timer = StepTimer()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            logits_processor=[timer],
        )
    total_s = time.perf_counter() - t0

    if timer.times:
        prefill_s = timer.times[0] - t0
        decode_times = [timer.times[i] - timer.times[i - 1] for i in range(1, len(timer.times))]
    else:
        prefill_s = total_s
        decode_times = []
    decode_total = sum(decode_times)
    prompt_len = inputs['input_ids'].shape[1]
    new_ids = out[0, prompt_len:].tolist()
    text = processor.tokenizer.decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    print('generate_total_s:', round(total_s, 4))
    print('prefill_s_to_first_logits:', round(prefill_s, 4))
    print('generated_tokens_total:', len(new_ids))
    print('decode_forward_steps:', len(decode_times))
    print('decode_total_s:', round(decode_total, 4))
    print('decode_avg_s_per_step:', round(decode_total / len(decode_times), 4) if decode_times else 0)
    print('decode_tokens_per_s:', round(len(decode_times) / decode_total, 4) if decode_total else 0)
    print('decode_step_times_s:', [round(x, 4) for x in decode_times])
    print('decoded_output:')
    print(text)

    result = {
        'name': name,
        'prefill_s': prefill_s,
        'decode_total_s': decode_total,
        'decode_steps': len(decode_times),
        'decode_avg_s_per_step': decode_total / len(decode_times) if decode_times else 0,
        'generate_total_s': total_s,
    }
    del model, processor, inputs, out
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--md', default=DEFAULT_MD)
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--dtype', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--order', choices=['orig-first', 'pruned-first'], default='orig-first')
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    with open(args.md, encoding='utf-8') as f:
        md_text = f.read()
    image = Image.open(args.image).convert('RGB')
    messages = split_rendered_md(md_text, args.image)

    order = [('orig', ORIG_MODEL), ('pruned', PRUNED_MODEL)]
    if args.order == 'pruned-first':
        order.reverse()

    results = [run_one(name, path, messages, image, args.max_new_tokens, dtype) for name, path in order]
    print('\n=== summary ===')
    for r in results:
        print(r['name'], 'generate_total_s=', round(r['generate_total_s'], 4), 'prefill_s=', round(r['prefill_s'], 4), 'decode_total_s=', round(r['decode_total_s'], 4), 'decode_steps=', r['decode_steps'], 'decode_avg_s_per_step=', round(r['decode_avg_s_per_step'], 4))


if __name__ == '__main__':
    main()
