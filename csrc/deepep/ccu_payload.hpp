#pragma once
#include <ATen/ATen.h>

namespace deep_ep {
// Transport only: no numeric conversion, arithmetic or scale interpretation.
inline bool IsCcuFp8(c10::ScalarType type) {
    return type == at::kFloat8_e4m3fn || type == at::kFloat8_e5m2 ||
           type == at::kFloat8_e4m3fnuz || type == at::kFloat8_e5m2fnuz;
}
inline bool IsCcuPayload(c10::ScalarType type) {
    return IsCcuFp8(type) || type == at::kHalf || type == at::kBFloat16 ||
           type == at::kFloat || type == at::kChar || type == at::kInt;
}
inline at::Tensor CcuByteView(const at::Tensor& tensor) {
    // FP8 elements are one byte; this view preserves rows, storage offset,
    // ownership and stream semantics without a cast or a staging allocation.
    return IsCcuFp8(tensor.scalar_type()) ? tensor.view(at::kByte) : tensor;
}
} // namespace deep_ep
