#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# Visual chunk 量化效果评估一键脚本
#
# 对照华为 CANN Kit 文档「量化效果评估」的插件方法:
#   fp chunk vs 量化 qmodel (optimize_model + calibrated pth),
#   在 256 个真实输入上对比输出余弦相似度 / MSE.
#
# 依赖 run_real_calib_256.sh 的产物:
#   - /temp/fdh/model_omc/model_visual_plugin_matmul_chunk{0..5}_real256/
#       dopt_config.chunk_0X.json
#       quant_output/calibrated_chunk_0X.pth
#   - /temp/fdh/input_calib/calib_inputs_256/  (真实 npz)
#   - HF 浮点模型: /temp/models/mobi0402_2B_halfimage_rl
# ============================================================

# ---- 环境 ----
source /opt/conda/etc/profile.d/conda.sh
conda activate cann
export DDK_DOPT=/temp/fdh/baiducloud/902137265_doulujiyao1/cann_codesample/cann_codesampe2_tar/cann_codesampe2/DDK-tools-next-6.0.1.0/tools/tools_dopt/dopt_pytorch_py3
REPO_ROOT=/home/ma-user/workspace/fdh/mobiinfer
export PYTHONPATH=${DDK_DOPT}:${REPO_ROOT}/transformers/llm/export:${PYTHONPATH:-}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 参数 (可改) ----
MODEL_PATH=${MODEL_PATH:-/temp/models/mobi0402_2B_halfimage_rl}
ROUTE_ROOT=${ROUTE_ROOT:-/temp/fdh/model_omc}
ROUTE_SUFFIX=${ROUTE_SUFFIX:-real256}
INPUT_DIR=${INPUT_DIR:-/temp/fdh/input_calib/calib_inputs_256}
OUT_DIR=${OUT_DIR:-/temp/fdh/input_calib/eval_real256}
NUM_SAMPLES=${NUM_SAMPLES:-256}
CHUNKS=${CHUNKS:-0,1,2,3,4,5}
NPU_CHUNKS=${NPU_CHUNKS:-6}

echo "=========================================="
echo "  Visual chunk 量化效果评估"
echo "  model      : ${MODEL_PATH}"
echo "  route_root : ${ROUTE_ROOT} (suffix=${ROUTE_SUFFIX})"
echo "  input_dir  : ${INPUT_DIR}"
echo "  out_dir    : ${OUT_DIR}"
echo "  chunks     : ${CHUNKS}  num_samples=${NUM_SAMPLES}"
echo "=========================================="

python3 eval_chunk_quant.py \
    --model_path "${MODEL_PATH}" \
    --route_root "${ROUTE_ROOT}" \
    --route_suffix "${ROUTE_SUFFIX}" \
    --input_dir "${INPUT_DIR}" \
    --out_dir "${OUT_DIR}" \
    --npu_chunks "${NPU_CHUNKS}" \
    --chunks "${CHUNKS}" \
    --num_samples "${NUM_SAMPLES}"
