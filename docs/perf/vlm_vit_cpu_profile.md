# VLM ViT CPU Profile Notes

This note records the minimum checks for profiling a VLM path where the ViT
encoder runs on ARM CPU with MNN.

## Confirm the loaded backend

First confirm that the app is loading the expected `libMNN.so`.

For the CPU-only Harmony build used for Kirin 9010 testing, the intended core
configuration is:

```text
MNN_NPU=OFF
MNN_KLEIDIAI=ON
MNN_KLEIDIAI_DEFAULT_ON=ON
MNN_ARM82=ON
MNN_SME2=OFF
MNN_SUPPORT_BF16=OFF
MNN_OPENCL=OFF
MNN_VULKAN=OFF
```

The dynamic dependency list should not contain HiAI libraries such as
`libhiai.so`, `libhiai_ir.so`, or `libhiai_ir_build.so`.

## Check KleidiAI usage

While running the ViT path, capture runtime logs from the host machine:

```bash
hdc hilog | grep -i -E "PROFILE|MNN|KleidiAI|MatMul|Attention"
```

Useful log strings:

```text
KleidiAI is running! AccelType is %s.
KleidiAI cannot accelerate! AccelType is %s.
```

Interpretation:

```text
KleidiAI is running
```

The current op shape and data layout hit a KleidiAI fast path.

```text
KleidiAI cannot accelerate
```

The op reached the KleidiAI selection code but did not match an accelerated
kernel. In that case, inspect the op type, tensor shape, quantization format,
and layout conversion around the ViT MatMul/Conv path.

No KleidiAI log usually means one of these:

- The app did not load a KleidiAI-enabled `libMNN.so`.
- The ViT path is not using CPU backend.
- The relevant ops are not routed through a KleidiAI-supported execution path.
- Logging is filtered out or disabled.

## Add coarse app-level timing

Separate ViT timing from preprocessing, projector, and LLM timing:

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
interpreter->runSession(visionSession);
auto t1 = std::chrono::high_resolution_clock::now();

auto ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
MNN_PRINT("PROFILE vit_runSession %.3f ms\n", ms);
```

Recommended profile checkpoints:

```text
PROFILE image_preprocess
PROFILE vit_runSession
PROFILE projector
PROFILE llm_prefill
PROFILE llm_decode_token
```

For ViT, also sweep CPU thread counts:

```text
1, 2, 4, 6, 8
```

On big.LITTLE SoCs, the best setting is often not the largest thread count.

## Build with MNN pipeline profile

For a diagnostic build, enable MNN's internal pipeline profile flag:

```bash
source ~/ohos-sdk/env.sh
cd /home/yydh/WAIC/mobiinfer/project/harmony/build

../build_64.sh \
  -DMNN_PIPELINE_PROFILE=ON \
  -DMNN_NPU:BOOL=OFF \
  -DMNN_KLEIDIAI:BOOL=ON \
  -DMNN_KLEIDIAI_DEFAULT_ON:BOOL=ON \
  -DKLEIDIAI_SRC_DIR=/home/yydh/env/kleidiai-1.16.0 \
  -DMNN_SME2:BOOL=OFF \
  -DMNN_SUPPORT_BF16:BOOL=OFF \
  -DMNN_OPENCL:BOOL=OFF \
  -DMNN_VULKAN:BOOL=OFF
```

Use this only for profiling. Do not keep `MNN_PIPELINE_PROFILE=ON` in the
release app build unless the extra logging/overhead is intentional.

## Use timeProfile when ViT is standalone

If the ViT encoder is available as a standalone `.mnn` model, run:

```bash
timeProfile.out vit.mnn 10 0 0 4
```

Then repeat for thread counts:

```bash
timeProfile.out vit.mnn 10 0 0 1
timeProfile.out vit.mnn 10 0 0 2
timeProfile.out vit.mnn 10 0 0 4
timeProfile.out vit.mnn 10 0 0 6
timeProfile.out vit.mnn 10 0 0 8
```

Focus on top-cost ops such as:

```text
MatMul / BatchMatMul
Attention
LayerNorm
Softmax
Transpose / Reshape / Gather
Resize / Normalize in preprocessing
```

## Common ViT CPU bottlenecks

- High image resolution increases patch/token count.
- MatMul or Conv does not hit KleidiAI fast paths.
- ViT remains fp32 while ARM82 fp16 paths are available.
- Layout conversion around attention/projector dominates runtime.
- Image resize/normalize/copy is counted as ViT time by mistake.
- Too many CPU threads cause scheduling overhead or thermal throttling.
