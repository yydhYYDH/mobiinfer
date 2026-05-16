#!/usr/bin/env python3
"""
Add MLP intermediate debug outputs to a visual block chunk model.

This inserts output points at every stage of the first block's MLP chain:
  post_attention_layernorm → fc1 → GELU → fc2 → residual_add

so that CPU vs NPU comparison reveals exactly which op introduces error.

Usage:
  1. Dump the .mnn to JSON:
     build_x86/MNNDump2Json visual_blocks_npu_1.mnn /tmp/vb1.json

  2. Add MLP outputs:
     python3 tools/add_mlp_outputs.py /tmp/vb1.json /tmp/vb1_mlp.json

  3. Convert back to .mnn:
     build_x86/MNNRevert2Buffer /tmp/vb1_mlp.json visual_blocks_npu_1_debug.mnn

  4. Push visual_blocks_npu_1_debug.mnn to phone (reuse same .mnn.weight file).
"""

import json
import sys
import os


# ---------------------------------------------------------------------------
# MLP intermediate points to add (for the FIRST block of the chunk, blocks.0)
# ---------------------------------------------------------------------------
MLP_DEBUG_TENSORS = [
    # (tensor name shown in trace, description)
    "/blocks.0/post_attention_layernorm/Add_1_output_0",   # LN before MLP
    "/blocks.0/mlp/linear_fc1/Add_output_0",               # fc1 output (before GELU)
    "/blocks.0/mlp/act_fn/Mul_5_output_0",                 # GELU output (before fc2)
    "/blocks.0/mlp/linear_fc2/Add_output_0",               # fc2 output (before residual)
]

# Optional: finer GELU internal checkpoints
GELU_EXTRA_TENSORS = [
    "/blocks.0/mlp/act_fn/Mul_3_output_0",    # input to Tanh
    "/blocks.0/mlp/act_fn/Tanh_output_0",     # Tanh output
]

# Always kept as-is
KEEP_OUTPUTS = [
    "hidden_states",
    "deepstack_hidden_0",
    "/blocks.0/Add_1_output_0",               # block 0 MLP residual
    "/blocks.0/self_attn/Add_output_0",       # block 0 attention residual
]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    json_in = sys.argv[1]
    json_out = sys.argv[2] if len(sys.argv) >= 3 else json_in.replace(".json", "_mlp.json")
    with_gelu_internals = "--gelu-internals" in sys.argv

    with open(json_in) as f:
        data = json.load(f)

    tensor_names = data.get("tensorName", [])
    original_outputs = data.get("outputName", [])

    # Build the new output list
    new_outputs = []

    # 1. Keep the standard final outputs first
    for name in KEEP_OUTPUTS:
        if name in tensor_names:
            if name not in new_outputs:
                new_outputs.append(name)
        else:
            # Fall back to original output order
            pass

    # 2. Add any remaining original outputs not already covered
    for name in original_outputs:
        if name not in new_outputs:
            new_outputs.append(name)

    # 3. Add MLP debug tensors
    for name in MLP_DEBUG_TENSORS:
        if name in tensor_names and name not in new_outputs:
            new_outputs.append(name)

    # 4. Optionally add GELU Tanh checkpoints
    if with_gelu_internals:
        for name in GELU_EXTRA_TENSORS:
            if name in tensor_names and name not in new_outputs:
                new_outputs.append(name)

    # Validate
    missing = [n for n in new_outputs if n not in tensor_names]
    if missing:
        print("ERROR: these tensor names not found in model:")
        for m in missing:
            print("  %s" % m)
        print("\nAvailable similar names:")
        for n in MLP_DEBUG_TENSORS + GELU_EXTRA_TENSORS + KEEP_OUTPUTS:
            block = n.split("/")[1] if "/" in n else ""
            for tn in tensor_names:
                if block and block in tn:
                    parts_n = set(n.split("/")[-2:])
                    parts_tn = set(tn.split("/")[-2:])
                    if parts_n & parts_tn:
                        print("  %s" % tn)
        sys.exit(1)

    data["outputName"] = new_outputs

    with open(json_out, "w") as f:
        json.dump(data, f, indent=2)

    # Print summary
    print("Modified model written to: %s" % json_out)
    print()
    print("Output index mapping (test will auto-label as deepstack_hidden_N):")
    print()
    labels = {
        "hidden_states":                                "final chunk output",
        "deepstack_hidden_0":                           "block 1 MLP residual",
        "/blocks.0/Add_1_output_0":                     "block 0 MLP residual",
        "/blocks.0/self_attn/Add_output_0":             "block 0 attention residual",
        "/blocks.0/post_attention_layernorm/Add_1_output_0": "LN before MLP",
        "/blocks.0/mlp/linear_fc1/Add_output_0":        "fc1 output (before GELU)",
        "/blocks.0/mlp/act_fn/Mul_5_output_0":          "GELU output (before fc2)",
        "/blocks.0/mlp/linear_fc2/Add_output_0":        "fc2 output (before residual)",
    }
    if with_gelu_internals:
        labels["/blocks.0/mlp/act_fn/Mul_3_output_0"] = "GELU: input to Tanh"
        labels["/blocks.0/mlp/act_fn/Tanh_output_0"] =  "GELU: Tanh output"

    for i, name in enumerate(new_outputs):
        desc = labels.get(name, "")
        print("  [%d] %-55s %s" % (i, name, desc))

    print()
    print("Diagnostic logic (test these in order):")
    print("  LN output   PASS -> gamma/beta loaded correctly")
    print("  LN output   FAIL -> LayerNorm gamma/beta broken on NPU")
    print("  fc1 output  FAIL (LN PASS) -> fc1 quantized weights wrong on NPU")
    print("  GELU output FAIL (fc1 PASS) -> GELU decomposition error (likely Tanh fp16)")
    print("  fc2 output  FAIL (GELU PASS) -> fc2 quantized weights wrong on NPU")
    print("  MLP residual FAIL (fc2 PASS) -> residual Add issue (unlikely)")


if __name__ == "__main__":
    main()
