#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# Generate OM files for all 6 visual chunks, calibrated with
# REAL image activations dumped from the MNN engine.
#
# Calibration inputs: ./calib_inputs_real  (real, from bin_to_chunk_npz.py)
# Route dirs:          ./model_visual_plugin_matmul_chunk{0..5}_real
# Platform:            kirin9020
#
# Quant config matches the smoke-tested chunk0 params exactly:
#   Quant_aigc_ptq / W8 min_max / A16 / input min_max unsigned
#   group_size=128 / use_qwen3_style_rotary
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- environment ---
source /opt/conda/etc/profile.d/conda.sh

conda activate cann
export DDK_DOPT=/data/fengdahu/cann_codesampe/DDK-tools-next-6.0.1.0/tools/tools_dopt/dopt_pytorch_py3
export PYTHONPATH=$DDK_DOPT:/home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export:${PYTHONPATH:-}

export DDK_DOPT=/temp/fdh/baiducloud/902137265_doulujiyao1/cann_codesample/cann_codesampe2_tar/cann_codesampe2/DDK-tools-next-6.0.1.0/tools/tools_dopt/dopt_pytorch_py3
export PYTHONPATH=$DDK_DOPT:/home/ma-user/workspace/fdh/mobiinfer/transformers/llm/export:$PYTHONPATH

# --- config (all match the chunk0 smoke test) ---
CALIB_INPUT_DIR=./calib_inputs_real
NPU_CHUNKS=6
NUM_SAMPLES=2
PLATFORM=kirin9020
SUFFIX=real

FAILED_CHUNKS=()

echo "=========================================="
echo "  all-6-chunks pipeline (REAL calibration)"
echo "  calib inputs: ${CALIB_INPUT_DIR}"
echo "  platform:     ${PLATFORM}"
echo "  num_samples:  ${NUM_SAMPLES}"
echo "=========================================="
echo ""

# sanity: calibration inputs must exist
FIRST_NPZ="${CALIB_INPUT_DIR}/chunk_00_sample_000.npz"
if [ ! -f "$FIRST_NPZ" ]; then
    echo "ERROR: calibration inputs not found: ${FIRST_NPZ}"
    echo "       run bin_to_chunk_npz.py first to generate ${CALIB_INPUT_DIR}"
    exit 1
fi
echo "[check] calibration inputs present: ${CALIB_INPUT_DIR}"
echo ""

# ---- per-chunk: all (prepare + calibrate + export-onnx) + OMC ----
for i in 0 1 2 3 4 5; do
    ROUTE_DIR="/temp/fdh/model_omc/model_visual_plugin_matmul_chunk_w8a8_${i}_${SUFFIX}"
    ONNX_FILE="${ROUTE_DIR}/onnx/visual_blocks_npu_${i}.onnx"
    # NOTE: run_visual_plugin_matmul_omc.sh uses --target=om, so the real
    # artifact is .om (the script's trailing ".omc" echo is misleading).
    OM_FILE="${ROUTE_DIR}/omc_output/visual_plugin_matmul_quantized.om"
    OMC_LOG="${ROUTE_DIR}/omc_output/omc_${PLATFORM}.log"

    echo "=========================================="
    echo "  chunk ${i} / 5"
    echo "  route: ${ROUTE_DIR}"
    echo "=========================================="

    # --- skip if final OM already exists ---
    if [ -f "$OM_FILE" ]; then
        echo "[chunk ${i}] OM already exists, skip: ${OM_FILE}"
        ls -lh "$OM_FILE" 2>/dev/null | awk '{print "  size: "$5}'
        echo ""
        continue
    fi

    # --- 1) all: prepare + calibrate + export-onnx ---
    echo "[chunk ${i}] running prepare + calibrate + export-onnx ..."
    if ! python visual_plugin_quant_matmul_route.py \
        --route_dir "$ROUTE_DIR" \
        --chunk_index $i \
        --npu_chunks "$NPU_CHUNKS" \
        --quant_strategy Quant_aigc_ptq \
        --weight_bit 8 \
        --weight_algo min_max \
        --act_bit 16 \
        --input_algo min_max \
        --num_samples "$NUM_SAMPLES" \
        --group_size 128 \
        --use_qwen3_style_rotary \
        --input_dir "$CALIB_INPUT_DIR" \
        --force_regen \
        all; then
        echo "[chunk ${i}] ERROR: all step failed"
        FAILED_CHUNKS+=("chunk_${i}:all")
        echo ""
        continue
    fi
    echo "[chunk ${i}] ONNX exported: ${ONNX_FILE}"

    # --- 2) OMC (-> .om) ---
    echo "[chunk ${i}] running OMG (${PLATFORM}) ..."
    if PLATFORM="$PLATFORM" \
        bash run_visual_plugin_matmul_omc.sh \
            "$ROUTE_DIR" \
            fp16 \
            > "$OMC_LOG" 2>&1; then
        if [ -f "$OM_FILE" ] && grep -q "OMG generate offline model success" "$OMC_LOG"; then
            echo "[chunk ${i}] OM SUCCESS: ${OM_FILE}"
            ls -lh "$OM_FILE" 2>/dev/null | awk '{print "  size: "$5}'
        else
            echo "[chunk ${i}] OMG ran but OM missing or success marker absent, check log: ${OMC_LOG}"
            FAILED_CHUNKS+=("chunk_${i}:omc_check")
        fi
    else
        echo "[chunk ${i}] OMG FAILED, check log: ${OMC_LOG}"
        FAILED_CHUNKS+=("chunk_${i}:omc")
    fi
    echo ""
done

# ---- summary ----
echo "=========================================="
echo "  pipeline finished"
echo "=========================================="

ALL_OK=1
for i in 0 1 2 3 4 5; do
    ROUTE_DIR="/temp/fdh/model_omc/model_visual_plugin_matmul_chunk_w8a8_${i}_${SUFFIX}"
    OM_FILE="${ROUTE_DIR}/omc_output/visual_plugin_matmul_quantized.om"
    if [ -f "$OM_FILE" ]; then
        SIZE=$(ls -lh "$OM_FILE" 2>/dev/null | awk '{print $5}')
        echo "  [OK]   chunk ${i}: ${OM_FILE}  (${SIZE})"
    else
        echo "  [FAIL] chunk ${i}: missing ${OM_FILE}"
        ALL_OK=0
    fi
done

echo ""
if [ ${#FAILED_CHUNKS[@]} -gt 0 ]; then
    echo "Failures: ${FAILED_CHUNKS[*]}"
    exit 1
fi
if [ $ALL_OK -ne 1 ]; then
    echo "Some OM files are missing."
    exit 1
fi

echo "All 6 OM files generated successfully with REAL calibration inputs."
