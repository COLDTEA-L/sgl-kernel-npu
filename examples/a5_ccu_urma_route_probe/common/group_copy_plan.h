#ifndef A5_CCU_GROUP_COPY_PLAN_H
#define A5_CCU_GROUP_COPY_PLAN_H
#include <array>
#include <cstdint>

namespace a5_ccu_urma_probe {
// Native HCCL GroupCopy geometry (CCU_MS_LOCAL_COPY_LOOP_COUNT=8,
// LOCAL_COPY_MS_PER_LOOP=8, CCU_MS_SIZE=4096). No route/port encoding here.
constexpr uint64_t GROUP_LANES = 8, GROUP_MS_BYTES = 4096, GROUP_MS_STRIDE = 8;
constexpr uint64_t GROUP_SLICE = GROUP_MS_BYTES * GROUP_MS_STRIDE;
constexpr uint64_t GROUP_BATCH = GROUP_SLICE * GROUP_LANES;
constexpr uint64_t GROUP_MAX_ITER = 8191;
constexpr uint64_t GROUP_MAX_BYTES = GROUP_BATCH * (GROUP_MAX_ITER + 1);

// Same parameter packing as HCCL ccu_kernel_utils.cc; verified against the
// installed old-object libhccl GetLoopParam/GetParallelParam/GetOffsetParam.
constexpr uint64_t GroupLoopParam(uint64_t step, uint64_t iterations)
{ return ((step & 0x1ffffffffULL) << 13) | (iterations & 0x3fffULL); }
constexpr uint64_t GroupParallelParam(uint64_t repeat, uint64_t index, uint64_t total)
{ return ((repeat & 0xffULL) << 55) | ((index & 0xffULL) << 48) | ((total & 0xffULL) << 41); }
constexpr uint64_t GroupOffsetParam()
{ return (GROUP_SLICE << 21) | (GROUP_MS_STRIDE << 10) | 1ULL; }

// Native CalGoSize layout: [bulk byte offset, repeated batches, tail parallel
// parameter, first tail slice bytes]. Tail is residual-first, then full slices.
inline bool BuildGroupCopyPlan(uint64_t bytes, std::array<uint64_t,4> *out)
{
    if (!out || !bytes || bytes > GROUP_MAX_BYTES) return false;
    uint64_t m=bytes/GROUP_BATCH, n=(bytes%GROUP_BATCH)/GROUP_SLICE, p=bytes%GROUP_SLICE;
    if (bytes==GROUP_MAX_BYTES) { m=GROUP_MAX_ITER; n=GROUP_LANES-1; p=GROUP_SLICE; }
    uint64_t parallel=0, residual=0;
    if (n && !p) { parallel=GroupParallelParam(n-1,0,1); residual=GROUP_SLICE; }
    else if (!n && p) { parallel=GroupParallelParam(0,0,1); residual=p; }
    else if (n && p) { parallel=GroupParallelParam(n-1,1,2); residual=p; }
    *out={{m*GROUP_BATCH,m,parallel,residual}};
    return true;
}
} // namespace a5_ccu_urma_probe
#endif
