#!/usr/bin/env python3
"""Trace the forward data-path of an MNN model from a given starting op.

Usage:
  1. Dump the .mnn to JSON:
     build_x86/MNNDump2Json model.mnn /tmp/model.json

  2. Trace main path only (default, shows side-branch names):
     python3 tools/trace_mnn_forward.py /tmp/model.json /blocks.0/Reshape_output_0

  3. Trace full tree with all branches:
     python3 tools/trace_mnn_forward.py /tmp/model.json /blocks.0/Reshape_output_0 --full

  4. Limit tree depth (default 6):
     python3 tools/trace_mnn_forward.py /tmp/model.json /blocks.0/Reshape_output_0 --full --depth 4

  5. List candidate start ops:
     python3 tools/trace_mnn_forward.py /tmp/model.json --list

  6. Show GELU decomposition:
     python3 tools/trace_mnn_forward.py /tmp/model.json --gelu blocks.0
"""

import json
import sys


def _build_indices(ops):
    """Return (op_by_out, consumers_by_tensor)."""
    op_by_out = {}
    consumers_by_tensor = {}
    for op in ops:
        for idx in op.get("outputIndexes", []):
            op_by_out[idx] = op
        for idx in op.get("inputIndexes", []):
            consumers_by_tensor.setdefault(idx, []).append(op)
    return op_by_out, consumers_by_tensor


def _tensor_label(tensor_idx, tensor_names):
    if tensor_idx < len(tensor_names) and tensor_names[tensor_idx]:
        return tensor_names[tensor_idx]
    return "t%d" % tensor_idx


# ---------------------------------------------------------------------------
# Main-path-only trace (original behaviour)
# ---------------------------------------------------------------------------
def trace_forward(data, start_op_name, max_depth=80):
    ops = data["oplists"]
    tensor_names = data.get("tensorName", [])
    output_names = data.get("outputName", [])

    start_out = None
    for op in ops:
        if op.get("name") == start_op_name:
            start_out = op["outputIndexes"][0] if op.get("outputIndexes") else None
            break

    if start_out is None:
        print("ERROR: op '%s' not found in oplists" % start_op_name)
        return

    current_out = start_out
    depth = 0
    while current_out is not None and depth < max_depth:
        consumers = [op for op in ops if current_out in op.get("inputIndexes", [])]
        if not consumers:
            break

        op = consumers[0]  # main data-path
        name = op.get("name", "?")
        mtype = op.get("main_type", "?")

        out_tensors = [
            tensor_names[i] for i in op.get("outputIndexes", [])
            if i < len(tensor_names)
        ]
        is_output = any(t in output_names for t in out_tensors)
        marker = "  *** OUTPUT ***" if is_output else ""

        # Show side branches (other consumers of the same tensor)
        if len(consumers) > 1:
            side_names = []
            for c in consumers[1:]:
                cn = c.get("name", "?")
                ct = c.get("main_type", "?")
                # Shorten: strip common prefix
                short = cn.split("/")[-1] if "/" in cn else cn
                side_names.append("%s(%s)" % (short, ct))
            print("[%3d] %-22s %s%s  [side branches: %s]" %
                  (depth, mtype, name, marker, ", ".join(side_names)))
        else:
            print("[%3d] %-22s %s%s" % (depth, mtype, name, marker))

        current_out = op.get("outputIndexes", [None])[0]
        depth += 1


# ---------------------------------------------------------------------------
# Full tree trace (shows all branches)
# ---------------------------------------------------------------------------
def trace_full(data, start_op_name, max_depth=6):
    ops = data["oplists"]
    tensor_names = data.get("tensorName", [])
    output_names = data.get("outputName", [])
    op_by_out, consumers_by_tensor = _build_indices(ops)

    # Locate the tensor produced by start_op_name
    start_tensor = None
    for op in ops:
        if op.get("name") == start_op_name:
            idxs = op.get("outputIndexes", [])
            if idxs:
                start_tensor = idxs[0]
            break

    if start_tensor is None:
        print("ERROR: op '%s' not found or has no output" % start_op_name)
        return

    visited = set()
    _print_tree_node(start_tensor, "", True, 0, max_depth,
                     tensor_names, output_names, op_by_out, consumers_by_tensor, visited)


def _print_tree_node(tensor_idx, prefix, is_last, depth, max_depth,
                     tensor_names, output_names, op_by_out, consumers_by_tensor, visited):
    """Recursively print a tensor and all its consumers."""
    label = _tensor_label(tensor_idx, tensor_names)
    is_out = label in output_names

    # Truncate label for display
    short_label = label
    if len(short_label) > 55:
        short_label = "..." + short_label[-52:]

    marker = " *** OUTPUT ***" if is_out else ""
    if is_out:
        marker = " *** OUTPUT ***"

    # Show the op that produced this tensor
    producer = op_by_out.get(tensor_idx)
    if producer is not None:
        ptype = producer.get("main_type", "?")
        pname = producer.get("name", "?")
        # Show additional inputs (side inputs for residual connections)
        side_inputs = []
        for inp in producer.get("inputIndexes", []):
            if inp != tensor_idx and inp not in visited:
                side_inputs.append(_tensor_label(inp, tensor_names))
        side_str = ""
        if side_inputs:
            side_str = "  [side-inputs: %s]" % ", ".join(s[:30] for s in side_inputs)
        print("%s%s %s → %s%s%s" % (prefix, "└─" if is_last else "├─",
                                    ptype, short_label, marker, side_str))
    else:
        print("%s%s (tensor) %s%s" % (prefix, "└─" if is_last else "├─",
                                      short_label, marker))

    if depth >= max_depth:
        return
    if tensor_idx in visited:
        if producer is not None:
            print(prefix + ("    " if is_last else "│   ") + "(cycle)")
        return
    visited.add(tensor_idx)

    consumers = consumers_by_tensor.get(tensor_idx, [])
    if not consumers:
        return

    for i, consumer in enumerate(consumers):
        is_last_child = (i == len(consumers) - 1)
        ctype = consumer.get("main_type", "?")
        cname = consumer.get("name", "?")
        short_cname = cname.split("/")[-1] if "/" in cname else cname

        child_prefix = prefix + ("    " if is_last else "│   ")
        conn = "└─" if is_last_child else "├─"
        print("%s%s[%s] %s" % (child_prefix, conn, ctype, short_cname[:55]))

        # Recurse into this consumer's output
        out_idxs = consumer.get("outputIndexes", [])
        grandchild_prefix = child_prefix + ("    " if is_last_child else "│   ")
        for j, out_idx in enumerate(out_idxs):
            is_last_gc = (j == len(out_idxs) - 1)
            _print_tree_node(out_idx, grandchild_prefix, is_last_gc,
                             depth + 1, max_depth,
                             tensor_names, output_names, op_by_out,
                             consumers_by_tensor, visited)


def list_boundary_ops(data):
    """Print potential start ops: block entry points, LayerNorms, etc."""
    ops = data["oplists"]
    output_names = data.get("outputName", [])

    print("=== Block entry reshapes ===")
    for op in ops:
        name = op.get("name", "")
        if "Reshape_output_0" in name and "self_attn" not in name and "input_layernorm" not in name:
            print("  %s" % name)

    print("\n=== LayerNorm ops ===")
    for op in ops:
        name = op.get("name", "")
        if op.get("main_type") == "LayerNorm":
            print("  %s" % name)

    print("\n=== Current model outputs ===")
    for name in output_names:
        print("  %s" % name)

    print("\n=== GELU act_fn ops (if any) ===")
    for op in ops:
        name = op.get("name", "")
        if "act_fn/" in name:
            print("  %-35s type=%s" % (name.split("/")[-1], op.get("main_type", "?")))


def show_gelu_decomposition(data, block_name="blocks.0"):
    """Print the GELU decomposition chain for a given block."""
    ops = data["oplists"]
    tensor_names = data.get("tensorName", [])

    prefix = "/%s/mlp/act_fn/" % block_name
    gelu_ops = [op for op in ops if op.get("name", "").startswith(prefix)]
    if not gelu_ops:
        print("No GELU ops found for block '%s'" % block_name)
        return

    gelu_ops.sort(key=lambda op: op.get("name", ""))
    print("GELU decomposition for /%s/mlp/act_fn/:" % block_name)
    for op in gelu_ops:
        name = op.get("name", "").split("/")[-1]
        mtype = op.get("main_type", "?")
        inputs = op.get("inputIndexes", [])
        input_info = []
        for inp in inputs:
            tn = tensor_names[inp] if inp < len(tensor_names) else str(inp)
            input_info.append(tn.split("/")[-1][:35] if "/" in tn else tn[:35])
        print("  %-30s type=%-16s inputs=%s" % (name, mtype, input_info))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    json_path = sys.argv[1]
    args = sys.argv[2:]

    # Parse optional flags
    full_mode = "--full" in args
    if full_mode:
        args.remove("--full")
    max_depth = None  # None = use each function's own default
    if "--depth" in args:
        idx = args.index("--depth")
        max_depth = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    with open(json_path) as f:
        data = json.load(f)

    if not args:
        trace_forward(data, "/blocks.0/Reshape_output_0")
        return

    if args[0] == "--list":
        list_boundary_ops(data)
    elif args[0] == "--gelu":
        block = args[1] if len(args) > 1 else "blocks.0"
        show_gelu_decomposition(data, block)
    elif full_mode:
        trace_full(data, args[0], max_depth=6 if max_depth is None else max_depth)
    else:
        if max_depth is not None:
            trace_forward(data, args[0], max_depth=max_depth)
        else:
            trace_forward(data, args[0])


if __name__ == "__main__":
    main()
