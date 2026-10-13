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

// Geometry sweep is restricted to the isolated local-copy diagnostic.
// Production callers omit this argument and retain native 8 x 32KiB.
struct GroupCopyConfig {
    uint64_t sliceBytes;
    constexpr explicit GroupCopyConfig(uint64_t slice = GROUP_SLICE) : sliceBytes(slice) {}
    bool Valid() const { return sliceBytes == 32768 || sliceBytes == 65536 || sliceBytes == 131072; }
    uint64_t MsStride() const { return sliceBytes / GROUP_MS_BYTES; }
    uint64_t BatchBytes() const { return sliceBytes * GROUP_LANES; }
};

// Same parameter packing as HCCL ccu_kernel_utils.cc; verified against the
// installed old-object libhccl GetLoopParam/GetParallelParam/GetOffsetParam.
constexpr uint64_t GroupLoopParam(uint64_t step, uint64_t iterations)
{ return ((step & 0x1ffffffffULL) << 13) | (iterations & 0x3fffULL); }
constexpr uint64_t GroupParallelParam(uint64_t repeat, uint64_t index, uint64_t total)
{ return ((repeat & 0xffULL) << 55) | ((index & 0xffULL) << 48) | ((total & 0xffULL) << 41); }
constexpr uint64_t GroupOffsetParam(uint64_t sliceBytes = GROUP_SLICE)
{ return (sliceBytes << 21) | ((sliceBytes / GROUP_MS_BYTES) << 10) | 1ULL; }

// Native CalGoSize layout: [bulk byte offset, repeated batches, tail parallel
// parameter, first tail slice bytes]. Tail is residual-first, then full slices.
inline bool BuildGroupCopyPlan(uint64_t bytes, std::array<uint64_t,4> *out,
    const GroupCopyConfig &config = GroupCopyConfig())
{
    if (!out || !config.Valid() || !bytes || bytes > GROUP_MAX_BYTES) return false;
    const uint64_t slice=config.sliceBytes, batch=config.BatchBytes();
    uint64_t m=bytes/batch, n=(bytes%batch)/slice, p=bytes%slice;
    if (m > GROUP_MAX_ITER) { m=GROUP_MAX_ITER; n=GROUP_LANES-1; p=slice; }
    uint64_t parallel=0, residual=0;
    if (n && !p) { parallel=GroupParallelParam(n-1,0,1); residual=slice; }
    else if (!n && p) { parallel=GroupParallelParam(0,0,1); residual=p; }
    else if (n && p) { parallel=GroupParallelParam(n-1,1,2); residual=p; }
    *out={{m*batch,m,parallel,residual}};
    return true;
}
} // namespace a5_ccu_urma_probe
#endif
