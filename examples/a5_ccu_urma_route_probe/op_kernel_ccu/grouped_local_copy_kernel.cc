#include "grouped_local_copy_kernel.h"
namespace a5_ccu_urma_probe {
namespace ccu = AscendC::ccu;
HcclResult GroupedLocalCopyKernel::GroupedLocalCopy(ccu::LocalAddr dst,
    ccu::LocalAddr src, std::array<ccu::Variable,4> go, const GroupCopyConfig& config)
{
    if (!config.Valid()) return HCCL_E_PARA;
    // Public arrays allocate contiguous MS/events; LoopGroup owns loop engines.
    ccu::Array<ccu::CcuBuffer> buffers(config.lanes * config.MsStride());
    ccu::Array<ccu::Event> events(config.lanes);
    std::array<ccu::LocalAddr, 2> bodySrc, bodyDst;
    std::array<ccu::Variable, 2> bodyBytes;
    auto set = [&](size_t i, const ccu::Variable& len) {
        bodySrc[i].addr = src.addr; bodySrc[i].token = src.token;
        bodyDst[i].addr = dst.addr; bodyDst[i].token = dst.token;
        bodyBytes[i] = len;
    };
    auto makeLoop = [&](size_t i, ccu::Variable& cfg) {
        ccu::Func body([&, i]() {
            CCU_THROW_IF_FAILED(ccu::LocalCopy(buffers[i * config.MsStride()], bodySrc[i], bodyBytes[i], events[i]), "self HBM to MS");
            CCU_THROW_IF_FAILED(ccu::EventWait(events[i]), "self MS read completion");
            CCU_THROW_IF_FAILED(ccu::LocalCopy(bodyDst[i], buffers[i * config.MsStride()], bodyBytes[i], events[i]), "self MS to HBM");
            CCU_THROW_IF_FAILED(ccu::EventWait(events[i]), "self HBM write completion");
        });
        return ccu::Loop(cfg, body);
    };
    // Define both bodies once at top level, as in the public SDK GroupCopy
    // example. The runtime configurations are assigned before each group.
    ccu::Variable cfg0, cfg1;
    auto first = makeLoop(0, cfg0), repeat = makeLoop(1, cfg1);
    ccu::Variable slice, offset;
    slice = config.sliceBytes; offset = GroupOffsetParam(config.sliceBytes);
    CCU_IF(go[0] != 0) {
        set(0, slice);
        ccu::Variable parallel;
        cfg0 = GroupLoopParam(config.BatchBytes(), 0); cfg0 += go[1];
        parallel = GroupParallelParam(config.lanes - 1, 0, 1);
        // maxLoopNum counts templates/AddLoop calls, not parallel MS lanes.
        ccu::LoopGroup group(parallel, offset, 1, {first});
    }
    CCU_IF(go[2] != 0) {
        src.addr += go[0]; dst.addr += go[0];
        set(0, go[3]);
        src.addr += go[3]; dst.addr += go[3];
        set(1, slice);
        cfg0 = GroupLoopParam(0, 1); cfg1 = GroupLoopParam(0, 1);
        auto parallel = go[2]; // copy construction aliases the variable handle
        ccu::LoopGroup group(parallel, offset, 2, {first, repeat});
    }
    return HCCL_SUCCESS;
}
} // namespace a5_ccu_urma_probe
