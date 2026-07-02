#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <route_dir> [fp16|fp32]"
    exit 1
fi

ROUTE_DIR=$(realpath "$1")
WEIGHT_DTYPE=${2:-fp16}
OMG_TOOL=${OMG_TOOL:-/temp/fdh/baiducloud/902137265_doulujiyao1/cann_codesample/cann_codesampe2_tar/cann_codesampe2/DDK-tools-next-6.0.1.0/tools/tools_omg/omg}
OMG_MASTER_DIR=${OMG_MASTER_DIR:-/temp/fdh/baiducloud/902137265_doulujiyao1/cann_codesample/cann_codesampe2_tar/cann_codesampe2/DDK-tools-next-6.0.1.0/tools/tools_omg/master}
PLATFORM=${PLATFORM:-kirinx90}
SAVE_WEIGHTS_AS_EXTERNAL_DATA=${SAVE_WEIGHTS_AS_EXTERNAL_DATA:-false}
PLATFORM_PLUGIN_DIR=$(dirname "${OMG_TOOL}")/../platform/${PLATFORM}

ROUTE_CONFIG=${ROUTE_DIR}/route_config.json
MODEL=$(python - <<'PY' "${ROUTE_CONFIG}"
import json, sys
cfg = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
print(cfg['model'])
PY
)
INPUT_SHAPE=$(python - <<'PY' "${ROUTE_CONFIG}"
import json, sys
cfg = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
print(cfg['input_shape'])
PY
)
QUANT_PARAMS=${ROUTE_DIR}/quant_output/quant_params_file
OUTPUT_PREFIX=${ROUTE_DIR}/omc_output/visual_plugin_matmul_quantized

mkdir -p "${ROUTE_DIR}/omc_output"
export PATH="${OMG_MASTER_DIR}:${PATH}"

omg_args=(
  --model "${MODEL}"
  --framework 5
  --output "${OUTPUT_PREFIX}"
  --input_shape="${INPUT_SHAPE}"
  --weight_data_type="${WEIGHT_DTYPE^^}"
  --save_weights_as_external_data="${SAVE_WEIGHTS_AS_EXTERNAL_DATA}"
  --target=om
  --compress_conf "${QUANT_PARAMS}"
)

if [ -n "${PLATFORM}" ] && [ -d "${PLATFORM_PLUGIN_DIR}" ]; then
  omg_args+=(--platform="${PLATFORM}")
else
  echo "WARN: platform plugin not found, skip --platform"
  echo "      expected: ${PLATFORM_PLUGIN_DIR}"
fi

"${OMG_TOOL}" "${omg_args[@]}"

echo "Done: ${OUTPUT_PREFIX}.omc"
