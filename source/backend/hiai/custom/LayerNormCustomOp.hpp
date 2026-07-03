#ifndef MNN_HIAI_CUSTOM_LAYERNORMCUSTOMOP_HPP
#define MNN_HIAI_CUSTOM_LAYERNORMCUSTOMOP_HPP

#include <graph/operator_hiai_reg.h>

namespace hiai {

HIAI_REG_OP(LayerNormCustom)
.HIAI_INPUT(x, TensorType({ALL}))
.HIAI_INPUT(gamma, TensorType({ALL}))
.HIAI_INPUT(beta, TensorType({ALL}))
.HIAI_OUTPUT(y, TensorType({ALL}))
.HIAI_REQUIRED_ATTR(epsilon, AttrValue::FLOAT)
.HIAI_REQUIRED_ATTR(norm_size, AttrValue::INT)
.HIAI_OP_END()

} // namespace hiai

#endif // MNN_HIAI_CUSTOM_LAYERNORMCUSTOMOP_HPP
