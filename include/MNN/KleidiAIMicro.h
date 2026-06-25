//
//  KleidiAIMicro.h
//  MNN
//
//  Created by MNN on 2026/06/25.
//

#ifndef MNN_KleidiAIMicro_h
#define MNN_KleidiAIMicro_h

#include <MNN/MNNDefine.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int m;
    int n;
    int k;
    int block_size;
    int iterations;
    uint64_t rhs_packed_bytes;
    uint64_t lhs_packed_bytes;
    double rhs_pack_ms;
    double lhs_pack_ms;
    double matmul_ms;
    double total_ms;
} MNNKleidiAIQ4F16Profile;

// Creates a QI4_ASYM_BLKQT_F16 micro-kernel context and pre-packs RHS weights.
// Shapes use C = LHS[M, K] * RHS[K, N], with RHS stored as N x K int4 data in the
// same packed source layout used by MNN low-memory int4 weights.
MNN_PUBLIC void* MNNKleidiAIQ4F16Create(int n, int k, int block_size, const uint8_t* weight_int4,
                                        const float* scale, const float* zero, const float* bias,
                                        MNNKleidiAIQ4F16Profile* profile);

// Runs LHS fp16 dynamic quant-pack and QI4_ASYM_BLKQT_F16 matmul. lhs_fp16 and dst_fp16 are raw fp16 buffers.
// Returns 0 on success, negative on error.
MNN_PUBLIC int MNNKleidiAIQ4F16Run(void* handle, const uint16_t* lhs_fp16, uint16_t* dst_fp16, int m,
                                   int iterations, MNNKleidiAIQ4F16Profile* profile);

MNN_PUBLIC void MNNKleidiAIQ4F16Destroy(void* handle);

#ifdef __cplusplus
}
#endif

#endif // MNN_KleidiAIMicro_h
