#pragma once
#include "ep8_relay_planner.hpp"
#include <ATen/ATen.h>

namespace deep_ep::ep8 {
using ChunkViews = std::array<at::Tensor, 4>;
inline std::pair<std::vector<at::Tensor>, std::vector<ChunkViews>> BindWorkspaceViews(
        const std::vector<at::Tensor>& arena, const std::array<U, 4>& sizes,
        const std::vector<std::array<U, 4>>& chunkSizes) {
    TORCH_CHECK(arena.size() == 8, "EP8 views: requires eight workspace tensors");
    for (size_t i = 0; i < 8; ++i) {
        const auto& t = arena[i];
        TORCH_CHECK(t.dim() >= 1 && t.is_contiguous() && sizes[i % 4] <= static_cast<U>(t.size(0)),
                    "EP8 views: workspace shape/capacity mismatch");
        TORCH_CHECK(t.device() == arena[0].device() && t.scalar_type() == arena[0].scalar_type() &&
                    t.sizes().slice(1) == arena[0].sizes().slice(1), "EP8 views: workspace types differ");
    }
    for (const auto& row : chunkSizes) for (size_t i = 0; i < 4; ++i)
        TORCH_CHECK(row[i] <= sizes[i], "EP8 views: chunk exceeds pool size");
    std::vector<at::Tensor> pool;
    pool.reserve(8);
    for (size_t i = 0; i < 8; ++i) pool.push_back(arena[i].narrow(0, 0, sizes[i % 4]));
    std::vector<ChunkViews> chunks(chunkSizes.size());
    for (size_t j = 0; j < chunkSizes.size(); ++j) for (size_t i = 0; i < 4; ++i)
        chunks[j][i] = pool[(j % 2) * 4 + i].narrow(0, 0, chunkSizes[j][i]);
    return {std::move(pool), std::move(chunks)};
}

// One Python/C++ boundary per pack/forward/unpack phase. Descriptors remain in
// C++; device copies use the caller's current stream, like the Python reference.
// This batches host submission, NOT the device kernels or their dependencies.
inline void CopySpansOut(const at::Tensor& src, at::Tensor dst, const CopySpans& copies) {
    TORCH_CHECK(src.dim() >= 1 && src.dim() == dst.dim(), "EP8 copy: invalid tensor rank");
    TORCH_CHECK(src.device() == dst.device() && src.scalar_type() == dst.scalar_type(),
                "EP8 copy: device/dtype mismatch");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous() && src.sizes().slice(1) == dst.sizes().slice(1),
                "EP8 copy: row shape/strides mismatch");
    // Validate ALL spans before submitting even the first copy.
    for (const auto& p : copies.spans) {
        TORCH_CHECK(p[0] <= static_cast<U>(src.size(0)) && p[2] <= static_cast<U>(src.size(0)) - p[0] &&
                    p[1] <= static_cast<U>(dst.size(0)) && p[2] <= static_cast<U>(dst.size(0)) - p[1],
                    "EP8 copy: span out of bounds");
    }
    if (copies.spans.empty()) return;
    if (src.numel() && dst.numel()) {
        uintptr_t a = reinterpret_cast<uintptr_t>(src.data_ptr());
        uintptr_t b = reinterpret_cast<uintptr_t>(dst.data_ptr());
        TORCH_CHECK(a <= b ? b - a >= src.nbytes() : a - b >= dst.nbytes(),
                    "EP8 copy: overlapping buffers");
    }
    for (const auto& p : copies.spans) if (p[2])
        dst.narrow(0, p[1], p[2]).copy_(src.narrow(0, p[0], p[2]));
}
} // namespace deep_ep::ep8
