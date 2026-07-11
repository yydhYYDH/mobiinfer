#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# Generate OM files for all 6 visual chunks with seq_len=608
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- environment ---
source /data/dahu/anaconda3/etc/profile.d/conda.sh
conda activate llama
export DDK_DOPT=/data/fengdahu/cann_codesampe/DDK-tools-next-6.0.1.0/tools/tools_dopt/dopt_pytorch_py3
export PYTHONPATH=$DDK_DOPT:/data/dahu/mlsys/MNN/transformers/llm/export:${PYTHONPATH:-}

MODEL_PATH=/data/fengdahu/model/mobi0429_2B__nore_halfimage
CALIB_DIR=./model_visual_plugin_matmul_calib_seq608
NPU_CHUNKS=6
NUM_SAMPLES=2
SEQ_LEN=608
PLATFORM=kirin9020

FAILED_CHUNKS=()

echo "=========================================="
echo "  seq_len=${SEQ_LEN}  all-6-chunks pipeline"
echo "  model: ${MODEL_PATH}"
echo "  platform: ${PLATFORM}"
echo "=========================================="
echo ""

# ---- step 1: generate seq_len=608 calibration inputs (once) ----
CALIB_INPUT_DIR="${CALIB_DIR}/calib_inputs"
FIRST_NPZ="${CALIB_INPUT_DIR}/chunk_00_sample_000.npz"

if [ -f "$FIRST_NPZ" ]; then
    echo "[step 1] calibration inputs already exist, skip: ${CALIB_INPUT_DIR}"
else
    echo "[step 1] generating seq_len=${SEQ_LEN} calibration inputs ..."
    python collect_visual_act_stats_v6.py \
        --path "$MODEL_PATH" \
        --dst_path "$CALIB_DIR" \
        --npu_chunks "$NPU_CHUNKS" \
        --num_samples 4 \
        --seq_len "$SEQ_LEN" \
        --dtype fp16
    echo "[step 1] done: ${CALIB_INPUT_DIR}"
fi
echo ""

# ---- step 2+3: per-chunk all + OMC ----
for i in 0 1 2 3 4 5; do
    ROUTE_DIR="./model_visual_plugin_matmul_chunk${i}_seq${SEQ_LEN}"
    ONNX_FILE="${ROUTE_DIR}/onnx/visual_blocks_npu_${i}.onnx"
    OMC_FILE="${ROUTE_DIR}/omc_output/visual_plugin_matmul_quantized.om"

    echo "=========================================="
    echo "  chunk ${i} / 5"
    echo "  route: ${ROUTE_DIR}"
    echo "=========================================="

    # --- 2a: all (prepare + calibrate + export-onnx) ---
    if [ -f "$OMC_FILE" ]; then
        echo "[chunk ${i}] OMC already exists, skip: ${OMC_FILE}"
        echo ""
        continue
    fi

    echo "[chunk ${i}] running prepare + calibrate + export-onnx ..."
    if ! python visual_plugin_quant_matmul_route.py \
        --route_dir "$ROUTE_DIR" \
        --chunk_index $i \
        --npu_chunks "$NPU_CHUNKS" \
        --input_dir "$CALIB_INPUT_DIR" \
        --quant_strategy Quant_aigc_ptq \
        --weight_bit 8 \
        --weight_algo min_max \
        --act_bit 8 \
        --input_algo min_max \
        --input_unsigned_quant \
        --num_samples "$NUM_SAMPLES" \
        --group_size 128 \
        --use_qwen3_style_rotary \
        --force_regen \
        all; then
        echo "[chunk ${i}] ERROR: all step failed"
        FAILED_CHUNKS+=("chunk_${i}:all")
        echo ""
        continue
    fi
    echo "[chunk ${i}] ONNX exported: ${ONNX_FILE}"

    # --- 2b: OMC ---
    echo "[chunk ${i}] running OMC (${PLATFORM}) ..."
    OMC_LOG="${ROUTE_DIR}/omc_output/omc_${PLATFORM}.log"
    if PLATFORM="$PLATFORM" \
        bash run_visual_plugin_matmul_omc.sh \
            "$ROUTE_DIR" \
            fp16 \
            > "$OMC_LOG" 2>&1; then
        if grep -q "OMG generate offline model success" "$OMC_LOG"; then
            echo "[chunk ${i}] OMC SUCCESS: ${OMC_FILE}"
            ls -lh "$OMC_FILE" 2>/dev/null | awk '{print "  size: "$5}'
        else
            echo "[chunk ${i}] OMC ran but output does not show success, check log: ${OMC_LOG}"
            FAILED_CHUNKS+=("chunk_${i}:omc_check")
        fi
    else
        echo "[chunk ${i}] OMC FAILED, check log: ${OMC_LOG}"
        FAILED_CHUNKS+=("chunk_${i}:omc")
    fi
    echo ""
done

# ---- summary ----
echo "=========================================="
echo "  pipeline finished"
echo "=========================================="

for i in 0 1 2 3 4 5; do
    ROUTE_DIR="./model_visual_plugin_matmul_chunk${i}_seq${SEQ_LEN}"
    OMC_FILE="${ROUTE_DIR}/omc_output/visual_plugin_matmul_quantized.omc"
    if [ -f "$OMC_FILE" ]; then
        SIZE=$(ls -lh "$OMC_FILE" 2>/dev/null | awk '{print $5}')
        echo "  [OK]  chunk ${i}: ${OMC_FILE}  (${SIZE})"
    else
        echo "  [FAIL] chunk ${i}: missing ${OMC_FILE}"
    fi
done

if [ ${#FAILED_CHUNKS[@]} -gt 0 ]; then
    echo ""
    echo "Failures: ${FAILED_CHUNKS[*]}"
    exit 1
fi

echo ""
echo "All 6 OM files generated successfully."
