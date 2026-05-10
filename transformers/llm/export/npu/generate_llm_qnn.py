import os
MOBIINFER_MNN = os.environ.get('MOBIINFER_MNN', '0') == '1'

if MOBIINFER_MNN:
    #!/usr/bin/python

    import sys
    import os
    import argparse
    import subprocess
    import json
    import shutil
    import time


    def _resolve_defaults(args):
        """Fill target-dependent defaults (model_file / out_dir_name / cache_path).

        Run after argparse so that explicit user overrides win, and so that the
        two targets land in disjoint cache/output dirs (avoids npu_postreat.json
        or qnn/ name collisions when both targets are run in sequence).
        """
        if args.model_file is None:
            args.model_file = (
                'visual_blocks.mnn' if args.target == 'visual_blocks' else 'llm.mnn'
            )
        if args.out_dir_name is None:
            args.out_dir_name = (
                'qnn_visual' if args.target == 'visual_blocks' else 'qnn'
            )
        if args.cache_path is None:
            args.cache_path = 'tmp_visual_blocks' if args.target == 'visual_blocks' else 'tmp_llm'
        return args


    def _stream_subprocess(cmd, cwd=None):
        process = subprocess.Popen(
            cmd, bufsize=1,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, shell=True, cwd=cwd,
        )
        for line in process.stdout:
            print(line, end='')
        process.wait()
        return process.returncode


    def makeIO(args):
        """Step1: invoke build/generateLlmIO to materialize testdir/<seq>/{input,output}.mnn.

        For target=llm we keep the original prefill+decode pair (chunk_size and 1).
        For target=visual_blocks the generator runs once at vis_seq_len.
        """
        exe = os.path.join(os.getcwd(), args.mnn_path, "generateLlmIO")
        output = os.path.join(args.cache_path, 'testdir')
        seq = args.vis_seq_len if args.target == 'visual_blocks' else args.chunk_size
        # generateLlmIO CLI: <modelDir> <outDir> <blocksize> <target> <model_file>
        cmd = "{exe} {model} {out} {seq} {target} {mfile}".format(
            exe=exe, model=args.model, out=output, seq=seq,
            target=args.target, mfile=args.model_file,
        )
        print("[Step1 cmd]", cmd)
        return _stream_subprocess(cmd)


    def seperate(args):
        """Step2: invoke build/compilefornpu to slice the model into NPU/CPU subgraphs.

        The qnn json fed in here selects the test shapes (testdir entries) and the
        output cache subdir. KVCACHE_SIZE_LIMIT only matters for LLM-style models
        that actually contain KV cache ops; visual_blocks omits it.
        """
        exe = os.path.join(os.getcwd(), args.mnn_path, "compilefornpu")
        model = os.path.join(os.getcwd(), args.model, args.model_file)
        print("model:", model)

        skips = []
        if args.target == 'llm' and args.llm_split_at:
            # Force-break the transformer-block subgraph at the listed block indices
            # by skipping each block's MLP-residual `Add_1` output. Each break
            # creates one extra NPU sub-.so. Needed when a single .so would exceed
            # the x86_64 PC32 ±2GB linker limit (R_X86_64_PC32 truncation against
            # crtstuff's .bss/.tm_clone_table).
            # NB: the names use the raw MNN op names (with `/` and `.`); the QNN
            # backend later sanitizes these by replacing `/.:` with `_` for the
            # generated graph0.cpp, so don't be misled by what you see there.
            for idx in [int(s) for s in args.llm_split_at.split(',') if s.strip()]:
                skips.append('/blocks.%d/Add_1_output_0' % idx)
        config = {
            "type": "QNN",
            "skips": skips,
            "testdir": [],
            "cache": args.out_dir_name,
        }
        if args.target == 'visual_blocks':
            # ViT runs at one fixed shape only.
            config['testdir'].append(os.path.join("testdir", '%d' % args.vis_seq_len))
        else:
            config['KVCACHE_SIZE_LIMIT'] = args.max_history_token
            config['testdir'].append(os.path.join("testdir", '1'))
            config['testdir'].append(os.path.join("testdir", '%d' % args.chunk_size))

        cache = os.path.join(os.getcwd(), args.cache_path)
        json_name = '%s.json' % args.out_dir_name
        with open(os.path.join(cache, json_name), 'w') as f:
            f.write(json.dumps(config, indent=4))

        out_mnn = '%s/%s' % (args.out_dir_name, args.model_file)
        cmd = '{exe} {model} {out} {jf}'.format(exe=exe, model=model, out=out_mnn, jf=json_name)
        print("[Step2 cmd]", cmd)
        return _stream_subprocess(cmd, cwd=cache)


    def compile_qnn(args):
        """Step3: hand the postreat json off to QNN SDK (npu_convert.py)."""
        exe = os.path.join(os.getcwd(), args.mnn_path, "..", "source", "backend", "qnn", "npu_convert.py")
        cache = os.path.join(os.getcwd(), args.cache_path)
        cmd = "python3 {exe} npu_postreat.json {soc} {dsp}".format(
            exe=exe, soc=args.soc_id, dsp=args.dsp_arch,
        )
        print("[Step3 cmd]", cmd)
        return _stream_subprocess(cmd, cwd=cache)


    def output_qnn(args):
        """Step4: relocate qnn[_visual]/ from cache to model dir, write runtime config."""
        src = os.path.join(args.cache_path, args.out_dir_name)
        dst = os.path.join(args.model, args.out_dir_name)
        if os.path.exists(dst):
            try:
                shutil.rmtree(dst)
            except PermissionError as e:
                raise RuntimeError(
                    f"Cannot remove existing {dst} (likely owned by another user): {e}\n"
                    f"Hint: rm it manually with appropriate privileges, e.g. "
                    f"`sudo rm -rf {dst}`, then re-run."
                )
        shutil.move(src, dst)

        if args.target == 'llm':
            config_npu = {
                "llm_model": "%s/%s" % (args.out_dir_name, args.model_file),
                "backend_type": "cpu",
                "thread_num": 1,
                "precision": "low",
                "chunk_limits": [args.chunk_size, 1],
                "memory": "low",
                "sampler_type": "penalty",
                "penalty": 1.1,
            }
            with open(os.path.join(args.model, "config_qnn.json"), 'w') as f:
                f.write(json.dumps(config_npu, indent=4))
        else:
            # visual_blocks runs as part of the omni vision pipeline; the runtime
            # config lives in the model's main config.json. Print a hint instead
            # of overwriting it to avoid clobbering unrelated fields.
            print("[hint] To use the QNN-compiled visual_blocks at runtime, update "
                  "%s/config.json:" % args.model)
            print('    "visual_blocks_model": "%s/%s",' % (args.out_dir_name, args.model_file))
            print('    "visual_blocks_backend_type": "qnn"')
        shutil.rmtree(args.cache_path)


    def convert(args):
        cache = os.path.join(os.getcwd(), args.cache_path)
        os.makedirs(cache, exist_ok=True)
        sta = time.time()
        print("Step1: Make IO  [target=%s, model=%s]" % (args.target, args.model_file))
        code = makeIO(args)
        end = time.time()
        print("Cost: ", end - sta, ' s')
        sta = end
        print("Step2: Seperate Model")
        seperate(args, model_name, ids)
        end = time.time()
        print("Cost: ", end - sta, ' s')
        sta = end
        print("Step3: Compile to QNN")
        compile_qnn(args)
        end = time.time()
        print("Cost: ", end - sta, ' s')
        print("Step4: Move result file to ", args.model)
        output_qnn(args)
        print("End")


    def main():
        parser = argparse.ArgumentParser(description='generate_llm_qnn', formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--model', type=str, required=True,
                            help='Model directory exported by llmexport.py')
        parser.add_argument('--soc_id', type=int, required=True,
                            help='Qualcomm SoC id, e.g. 57 for 8 Gen3')
        parser.add_argument('--dsp_arch', type=str, required=True,
                            help='HTP DSP arch, e.g. v75 for 8 Gen3')
        parser.add_argument('--mnn_path', type=str, default="../../../build/",
                            help='Directory containing built generateLlmIO and compilefornpu')
        parser.add_argument('--cache_path', type=str, default=None,
                            help='Cache dir for intermediate files; default: tmp_<target>')
        parser.add_argument('--target', choices=['llm', 'visual_blocks'], default='llm',
                            help='Which sub-model to compile to QNN')
        parser.add_argument('--model_file', type=str, default=None,
                            help='.mnn filename inside --model dir; default by target '
                                 '(llm.mnn / visual_blocks.mnn)')
        parser.add_argument('--out_dir_name', type=str, default=None,
                            help='Output subdir under --model; default by target '
                                 '(qnn / qnn_visual)')
        parser.add_argument('--chunk_size', type=int, default=128,
                            help='[target=llm] prefill chunk_size')
        parser.add_argument('--vis_seq_len', type=int, default=676,
                            help='[target=visual_blocks] fixed seq_len; '
                                 'default 676 matches image_size=420 with patch=16, merge=2')
        parser.add_argument('--max_history_token', type=int, default=0,
                            help='[target=llm] KV cache history limit; 0 = unlimited')
        parser.add_argument('--llm_split_at', type=str, default='',
                            help='[target=llm] comma-separated transformer block indices to '
                                 'CHOP OFF UPSTREAM at (NOT a chunked split). Skipping '
                                 '/blocks.X/Add_1_output_0 makes compilefornpu drop blocks 0..X '
                                 'from the NPU graph entirely. Use only if you intentionally '
                                 'want a partial-model NPU bin. For "single .so > 2GB" issues, '
                                 'patch the QNN Makefile to objcopy --rename-section .data=.ldata '
                                 '(see npu_convert.py QNN_MIRROR support) instead.')
        args = parser.parse_args()
        args = _resolve_defaults(args)
        convert(args)


    if __name__ == '__main__':
        main()
else:
    #!/usr/bin/python

    import sys
    import os
    import argparse
    import subprocess
    import json
    import shutil
    import time

    def makeIO(args, model_name, inputjson, external_file = None):
        exe = os.path.join(os.getcwd(), args.mnn_path, "generateIO")
        model = os.path.join(os.getcwd(), args.model, model_name)
        cache = os.path.join(os.getcwd(), args.cache_path)
        output = os.path.join(cache, 'testdir')
        os.makedirs(output, exist_ok=True)
        print(os.popen(exe + " " + model + " " + inputjson + " " + output + " " + external_file).read())

    def makeIOJson(args, seq_len, hidden_size, mask_type):
        config = {
            "configs": [
                {
                    "inputs": [
                        {
                            "name": "input_ids",
                            "shape": [seq_len, 1, hidden_size]
                        },
                        {
                            "name": "attention_mask",
                            "shape": [1, 1, seq_len, seq_len],
                            "type": mask_type
                        },
                        {
                            "name": "position_ids",
                            "shape": [1, seq_len],
                            "type": "int"
                        },
                        {
                            "name": "logits_index",
                            "shape": [1],
                            "type": "int",
                            "value": 0
                        }
                    ],
                    "outputs": [
                        "logits"
                    ]
                },
                {
                    "inputs": [
                        {
                            "name": "input_ids",
                            "shape": [1, 1, hidden_size]
                        },
                        {
                            "name": "attention_mask",
                            "shape": [1, 1, 1, 1],
                            "type": mask_type
                        },
                        {
                            "name": "position_ids",
                            "shape": [1, 1],
                            "type": "int"
                        },
                        {
                            "name": "logits_index",
                            "shape": [1],
                            "type": "int",
                            "value": -1
                        }
                    ],
                    "outputs": [
                        "logits"
                    ]
                }
            ]
        }
        if "Qwen3.5" in args.model:
            cfg = config["configs"]
            inputs = cfg[0]["inputs"]
            for inp in inputs:
                if inp["name"] == "attention_mask":
                    inp["shape"] = [2, 1, seq_len, seq_len, 3]
                if inp["name"] == "position_ids":
                    inp["shape"] = [3, seq_len]

            inputs = cfg[1]["inputs"]
            for inp in inputs:
                if inp["name"] == "attention_mask":
                    inp["shape"] = [2, 1, 1, 1, 3]		
                if inp["name"] == "position_ids":
                    inp["shape"] = [3, 1]
        if "Qwen" in args.model and "VL" in args.model:
            cfg = config["configs"]
            inputs = cfg[0]["inputs"]
            for inp in inputs:
                if inp["name"] == "position_ids":
                    inp["shape"] = [3, seq_len]

            new_input = {
                "name": "deepstack_embeds",
                "shape": [3, 1, 1]
            }
            inputs.append(new_input)

            inputs = cfg[1]["inputs"]
            for inp in inputs:
                if inp["name"] == "position_ids":
                    inp["shape"] = [3, 1]

            new_input = {
                "name": "deepstack_embeds",
                "shape": [3, 1, 1]
            }
            inputs.append(new_input)
        cache = os.path.join(os.getcwd(), args.cache_path)
        with open(os.path.join(cache, 'input.json'), 'w') as f:
            f.write(json.dumps(config, indent=4))

    def makeVLIOJson(args, image_sizes):
        configs = []
        for w, h in image_sizes:
            if "Qwen2.5" in args.model and "VL" in args.model:
                align_size = 28
                grid_h = (round(h / align_size) * align_size) // 14
                grid_w = (round(w / align_size) * align_size) // 14
                seq_len = grid_h * grid_w
                config = {
                    "inputs": [
                        {"name": "patches", "shape": [seq_len, 1176]},
                        {"name": "position_ids", "shape": [2, seq_len]},
                        {"name": "attention_mask", "shape": [2, 1, seq_len, seq_len]},
                        {"name": "window_index", "shape": [seq_len//4]}
                    ],
                    "outputs": ["image_embeds"]
                }
            elif "Qwen3" in args.model or "Qwen3.5" in args.model:
                align_size = 32
                grid_h = (round(h / align_size) * align_size) // 16
                grid_w = (round(w / align_size) * align_size) // 16
                seq_len = grid_h * grid_w
                config = {
                    "inputs": [
                        {"name": "patches", "shape": [seq_len, 1536]},
                        {"name": "position_ids", "shape": [2, seq_len]},
                        {"name": "attention_mask", "shape": [1, seq_len, seq_len]},
                        {"name": "idx_tensor", "shape": [4, seq_len]},
                        {"name": "weight_tensor", "shape": [4, seq_len]}
                    ],
                    "outputs": ["image_embeds"]
                }
            elif "FastVLM" in args.model:
                config = {
                    "inputs": [
                        {"name": "input_images", "shape": [1, 3, h, w]}
                    ],
                    "outputs": ["image_embeds"]
                }
            else:
                raise ValueError(f"Unsupported visual model: {args.model}")
            configs.append(config)

        full_config = {"configs": configs}
        cache = os.path.join(os.getcwd(), args.cache_path)
        with open(os.path.join(cache, 'input.json'), 'w') as f:
            json.dump(full_config, f, indent=4)

    def convert_fastvlm(args, image_sizes):
        qnn_sdk = os.environ["QNN_SDK_ROOT"]
        exe = os.path.join(os.getcwd(), args.mnn_path, "MNN2QNNModel")
        model = os.path.join(os.getcwd(), args.model, 'visual.mnn')
        cache = os.path.join(os.getcwd(), args.cache_path)
        output = os.path.join(cache, 'qnn')
        os.makedirs(output, exist_ok=True)
        result = " ".join([f"1x3x{w}x{h}" for w, h in image_sizes])
        print(os.popen(exe + " " + qnn_sdk + " " + str(args.soc_id) + " " + str(args.dsp_arch).lstrip('v') + " " + model + " " + output + " " + str(len(image_sizes)) + " " + result).read())

        for item in os.listdir(output):
            s = os.path.join(output, item)
            d = os.path.join(args.model, item)
            if os.path.exists(d):
                if os.path.isfile(d): os.remove(d)
                else: shutil.rmtree(d)
            shutil.move(s, d)
        qnn_file = 'visual_' + str(args.soc_id) + "_" + str(args.dsp_arch).lstrip('v') + '.mnn'
        config_npu = {
            "llm_model": "llm.mnn",
            "llm_weight": "llm.mnn.weight",
            "backend_type": "cpu",
            "thread_num": 4,
            "precision": "low",
            "memory": "low",
            "sampler_type": "penalty",
            "penalty": 1.1,
            "visual_model": qnn_file,
             "mllm": {
                "backend_type": "cpu",
                "thread_num": 4,
                "precision": "normal",
                "memory": "low"
            }
        }
        with open(os.path.join(args.model, "config_qnn.json"), 'w') as f:
            f.write(json.dumps(config_npu, indent = 4))
        shutil.rmtree(args.cache_path)


    def seperate(args, model_name, ids):
        exe = os.path.join(os.getcwd(), args.mnn_path, "compilefornpu")
        model = os.path.join(os.getcwd(), args.model, model_name)
        config = {
            "type": "QNN",
            "skips": skips,
            "testdir": [],
            "cache": args.out_dir_name,
        }
        for i in ids:
            config['testdir'].append(os.path.join("testdir", '%d' %i))
        cache = os.path.join(os.getcwd(), args.cache_path)
        json_name = '%s.json' % args.out_dir_name
        with open(os.path.join(cache, json_name), 'w') as f:
            f.write(json.dumps(config, indent=4))
        process = subprocess.Popen(exe + ' ' + model + ' qnn/' + model_name + ' qnn.json', bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd = cache, text=True, shell=True)
        for line in process.stdout:
            print(line, end='')


    def compile_qnn(args):
        """Step3: hand the postreat json off to QNN SDK (npu_convert.py)."""
        exe = os.path.join(os.getcwd(), args.mnn_path, "..", "source", "backend", "qnn", "npu_convert.py")
        cache = os.path.join(os.getcwd(), args.cache_path)
        cmd = "python3 {exe} npu_postreat.json {soc} {dsp}".format(
            exe=exe, soc=args.soc_id, dsp=args.dsp_arch,
        )
        print("[Step3 cmd]", cmd)
        return _stream_subprocess(cmd, cwd=cache)


    def output_qnn(args):
        if os.path.exists(os.path.join(args.model, 'qnn')):
            shutil.rmtree(os.path.join(args.model, 'qnn'))
        shutil.move(os.path.join(args.cache_path, 'qnn'), os.path.join(args.model, 'qnn'))
        if args.need_config_json is True:
            config_path = os.path.join(args.model, 'config.json')
            config_npu = {}
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_npu = json.load(f)
            is_visual = args.model_name == "visual.mnn"
            if not is_visual:
                config_npu["llm_model"] = "qnn/llm.mnn"
                config_npu["chunk_limits"] = [args.chunk_size, 1]
            else:
                config_npu["visual_model"] = "qnn/visual.mnn"
            with open(os.path.join(args.model, "config_qnn.json"), 'w') as f:
                f.write(json.dumps(config_npu, indent = 4))
        shutil.rmtree(args.cache_path)

    def convert_qnn(args, model_name, inputjson, external_file, ids):
        sta = time.time()
        print("Step1: Make IO")
        makeIO(args, model_name, inputjson, external_file)
        end = time.time()
        print("Cost: ", end - sta, ' s')
        sta = end
        print("Step2: Seperate Model")
        seperate(args, model_name, ids)
        end = time.time()
        print("Cost: ", end - sta, ' s')
        sta = end
        print("Step3: Compile to QNN")
        compile_qnn(args)
        end = time.time()
        print("Cost: ", end - sta, ' s')
        print("Step4: Move result file to ", args.model)
        output_qnn(args)
        print("End")

    def convert_visual(args):
        cache = os.path.join(os.getcwd(), args.cache_path)
        os.makedirs(cache, exist_ok=True)

        external_file = os.path.join(os.getcwd(), args.model, 'visual.mnn.weight')
        image_sizes = []
        try:
            size_strs = [s.strip() for s in args.image_sizes.split(',')]
            for sz in size_strs:
                if 'x' not in sz:
                    raise ValueError(f"Invalid size format: {sz}, expected 'WxH'")
                w_str, h_str = sz.split('x')
                w, h = int(w_str), int(h_str)
                if w <= 0 or h <= 0:
                    raise ValueError(f"Width and height must be positive: {sz}")
                image_sizes.append((w, h))
        except Exception as e:
            print(f"Error parsing --image_sizes: {e}")
            sys.exit(1)

        if not image_sizes:
            print("No valid image sizes provided.")
            sys.exit(1)
        if "FastVLM" in args.model:
            convert_fastvlm(args, image_sizes)
        else:
            makeVLIOJson(args, image_sizes)
            inputjson = os.path.join(cache, 'input.json')
            ids = list(range(len(image_sizes)))
            convert_qnn(args, 'visual.mnn', inputjson, external_file, ids)

    def convert_llm(args):
        cache = os.path.join(os.getcwd(), args.cache_path)
        os.makedirs(cache, exist_ok=True)
        hidden_size = 768
        mask_type = "int"
        config_file_path = os.path.join(os.getcwd(), args.model, 'llm_config.json')
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            if "hidden_size" in config_data:
                hidden_size = config_data["hidden_size"]
            else:
                print(f"Error: 'hidden_size' key not found in {config_file_path}")
                return npu_convert
            if "attention_mask" in config_data:
                mask_type = config_data["attention_mask"]

        ids = [0, 1]
        external_file = os.path.join(os.getcwd(), args.model, 'llm.mnn.weight')
        makeIOJson(args, 128, hidden_size, mask_type)
        inputjson = os.path.join(cache, 'input.json')
        convert_qnn(args, 'llm.mnn', inputjson, external_file, ids)

    def convert_input_json(args):
        cache = os.path.join(os.getcwd(), args.cache_path)
        os.makedirs(cache, exist_ok=True)
        input_shape_num = 1
        with open(args.input_json, 'r') as f:
            data = json.load(f)
            if 'configs' in data and isinstance(data['configs'], list):
                input_shape_num = len(data['configs'])
                print(input_shape_num)

        ids = list(range(input_shape_num))
        args.need_config_json = False
        external_file = os.path.join(os.getcwd(), args.model, args.external_file)
        convert_qnn(args, args.model_name, args.input_json, external_file, ids)

    def convert(args):
        if args.input_json != "":
            convert_input_json(args)
        elif args.model_name == "llm.mnn":
            convert_llm(args)
        elif args.model_name == "visual.mnn":
            convert_visual(args)

    def main():
        parser = argparse.ArgumentParser(description='generate_llm_qnn', formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--model', type=str, required=True,
                            help='Model directory exported by llmexport.py')
        parser.add_argument('--soc_id', type=int, required=True,
                            help='Qualcomm SoC id, e.g. 57 for 8 Gen3')
        parser.add_argument('--dsp_arch', type=str, required=True,
                            help='HTP DSP arch, e.g. v75 for 8 Gen3')
        parser.add_argument('--mnn_path', type=str, default="../../../build/",
                            help='Directory containing built generateLlmIO and compilefornpu')
        parser.add_argument('--cache_path', type=str, default=None,
                            help='Cache dir for intermediate files; default: tmp_<target>')
        parser.add_argument('--target', choices=['llm', 'visual_blocks'], default='llm',
                            help='Which sub-model to compile to QNN')
        parser.add_argument('--model_file', type=str, default=None,
                            help='.mnn filename inside --model dir; default by target '
                                 '(llm.mnn / visual_blocks.mnn)')
        parser.add_argument('--out_dir_name', type=str, default=None,
                            help='Output subdir under --model; default by target '
                                 '(qnn / qnn_visual)')
        parser.add_argument('--chunk_size', type=int, default=128,
                            help='[target=llm] prefill chunk_size')
        parser.add_argument('--vis_seq_len', type=int, default=676,
                            help='[target=visual_blocks] fixed seq_len; '
                                 'default 676 matches image_size=420 with patch=16, merge=2')
        parser.add_argument('--max_history_token', type=int, default=0,
                            help='max history token, default is 0, which mean no limit for history token number'
        )
        parser.add_argument('--image_sizes', type=str, default="512x512",
                            help='Image sizes for vision model, e.g., "512x512" or "224x224,384x384,512x512"'
                            )
        parser.add_argument('--input_json', type=str, default="",
                            help='input json contain all input shape'
                            )
        parser.add_argument('--external_file', type=str, default="",
                            help='external file stored weight'
                            )
        parser.add_argument('--model_name', type=str, default="llm.mnn",
                            help='the name of model, like llm.mnn or visual.mnn'
                            )
        parser.add_argument('--need_config_json', type=bool, default=True,
                            help='wheather generate config json'
                            )
        args = parser.parse_args()
        args = _resolve_defaults(args)
        convert(args)


    if __name__ == '__main__':
        main()