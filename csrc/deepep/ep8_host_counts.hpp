#pragma once
#include "ep8_relay_planner.hpp"
#include <ATen/ATen.h>

namespace deep_ep::ep8 {
// Caller must complete D2H before entry. Copy the 64 values immediately; never
// retain the borrowed pinned buffer or implicitly synchronize a device tensor.
inline Matrix HostCountTensor(const at::Tensor& counts) {
    TORCH_CHECK(counts.device().is_cpu() && counts.scalar_type()==at::kLong &&
                counts.layout()==at::kStrided && counts.is_contiguous() &&
                ((counts.dim()==1 && counts.size(0)==64) ||
                 (counts.dim()==2 && counts.size(0)==8 && counts.size(1)==8)),
                "counts must be contiguous CPU int64[64] or int64[8,8]; finish D2H first");
    const auto* src=counts.data_ptr<int64_t>();
    Matrix out{};
    for (size_t i=0;i<out.size();++i) {
        TORCH_CHECK(src[i]>=0,"counts must be nonnegative");
        out[i]=static_cast<U>(src[i]);
    }
    return out;
}
}
