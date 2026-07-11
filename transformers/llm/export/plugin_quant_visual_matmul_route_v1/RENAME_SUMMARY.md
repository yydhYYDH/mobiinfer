# OM File Renaming Summary

## Execution Date
2026-06-20

## Script Used
`rename_om_files.sh`

## Renamed Files

All `visual_plugin_matmul_quantized.om` files in the `model_visual_plugin_matmul_chunk*_real/omc_output/` directories have been successfully renamed to `visual_blocks_npu_*.om` format:

| Original Directory | Old Filename | New Filename | Size |
|-------------------|--------------|--------------|------|
| model_visual_plugin_matmul_chunk0_real/omc_output/ | visual_plugin_matmul_quantized.om | **visual_blocks_npu_0.om** | ~49.9 MB |
| model_visual_plugin_matmul_chunk1_real/omc_output/ | visual_plugin_matmul_quantized.om | **visual_blocks_npu_1.om** | ~49.9 MB |
| model_visual_plugin_matmul_chunk2_real/omc_output/ | visual_plugin_matmul_quantized.om | **visual_blocks_npu_2.om** | ~49.9 MB |
| model_visual_plugin_matmul_chunk3_real/omc_output/ | visual_plugin_matmul_quantized.om | **visual_blocks_npu_3.om** | ~49.9 MB |
| model_visual_plugin_matmul_chunk4_real/omc_output/ | visual_plugin_matmul_quantized.om | **visual_blocks_npu_4.om** | ~49.9 MB |
| model_visual_plugin_matmul_chunk5_real/omc_output/ | visual_plugin_matmul_quantized.om | **visual_blocks_npu_5.om** | ~49.9 MB |

## Total Files Renamed
✅ **6 files** successfully renamed

## Verification
All renamed files have been verified and are present in their respective directories with correct naming convention.

## Notes
- The script automatically extracts chunk numbers from directory names
- Each file maintains its original size (~49.9 MB)
- Log files (`omc_kirin9020.log`) remain unchanged in each directory
- The script can be re-run safely (it will skip already renamed files)
