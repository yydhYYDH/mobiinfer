#!/bin/bash

# Script to rename visual_plugin_matmul_quantized.om files to visual_blocks_npu_*.om
# in model_visual_plugin_matmul_chunk*_real directories

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting OM file renaming process..."
echo "Working directory: $SCRIPT_DIR"
echo ""

# Counter for tracking renamed files
renamed_count=0
error_count=0

# Find all model_visual_plugin_matmul_chunk*_real directories
for chunk_dir in "$SCRIPT_DIR"/model_visual_plugin_matmul_chunk*_real; do
    # Check if directory exists
    if [ ! -d "$chunk_dir" ]; then
        continue
    fi

    # Extract chunk number from directory name
    dir_name=$(basename "$chunk_dir")
    if [[ $dir_name =~ model_visual_plugin_matmul_chunk([0-9]+)_real ]]; then
        chunk_num="${BASH_REMATCH[1]}"
    else
        echo "Warning: Could not extract chunk number from $dir_name, skipping..."
        continue
    fi

    # Define source and target paths
    omc_output_dir="$chunk_dir/omc_output"
    source_file="$omc_output_dir/visual_plugin_matmul_quantized.om"
    target_file="$omc_output_dir/visual_blocks_npu_${chunk_num}.om"

    # Check if source file exists
    if [ ! -f "$source_file" ]; then
        echo "Warning: Source file not found: $source_file"
        ((error_count++))
        continue
    fi

    # Rename the file
    if mv "$source_file" "$target_file"; then
        echo "✓ Renamed: $dir_name/omc_output/"
        echo "  From: visual_plugin_matmul_quantized.om"
        echo "  To:   visual_blocks_npu_${chunk_num}.om"
        ((renamed_count++))
    else
        echo "✗ Error renaming file in $dir_name"
        ((error_count++))
    fi
    echo ""
done

# Summary
echo "======================================"
echo "Renaming Complete!"
echo "======================================"
echo "Successfully renamed: $renamed_count files"
if [ $error_count -gt 0 ]; then
    echo "Errors encountered: $error_count"
fi
echo ""

# List all renamed files
if [ $renamed_count -gt 0 ]; then
    echo "Renamed files:"
    for chunk_dir in "$SCRIPT_DIR"/model_visual_plugin_matmul_chunk*_real; do
        if [ -d "$chunk_dir" ]; then
            dir_name=$(basename "$chunk_dir")
            if [[ $dir_name =~ model_visual_plugin_matmul_chunk([0-9]+)_real ]]; then
                chunk_num="${BASH_REMATCH[1]}"
                target_file="$chunk_dir/omc_output/visual_blocks_npu_${chunk_num}.om"
                if [ -f "$target_file" ]; then
                    file_size=$(du -h "$target_file" | cut -f1)
                    echo "  - $dir_name/omc_output/visual_blocks_npu_${chunk_num}.om ($file_size)"
                fi
            fi
        fi
    done
fi
