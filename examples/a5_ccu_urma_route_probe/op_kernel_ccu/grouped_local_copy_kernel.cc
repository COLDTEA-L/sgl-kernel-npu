// Implements the native GroupCopy structure with the target SDK's old object
// API. References: HCCL ccu_kernel_alg_base.cc CreateMultiOpCopy/GroupCopy,
// and installed old-object CcuKernelAlgBase::GroupCopy/LoopGroup. No new
// CcuWriteMemToMem/HcommCcuKernelRegister primitive ABI or private HCCL class.
#include "grouped_local_copy_kernel.h"
#include <string>
namespace a5_ccu_urma_probe {
namespace CcuRep = hcomm::CcuRep;
#define GROUP_CHECK(call) do { const auto s=(call); if(s!=HCCL_SUCCESS) return s; } while(0)
HcclResult GroupedLocalCopyKernel::GroupedLocalCopy(CcuRep::LocalAddr dst,
    CcuRep::LocalAddr src, const std::array<CcuRep::Variable,4> &go, const GroupCopyConfig &config)
{
    if (!config.Valid()) return HCCL_E_PARA;
    const uint64_t msStride=config.MsStride();
    moConfig_={static_cast<uint32_t>(msStride),static_cast<uint32_t>(config.lanes),config.sliceBytes};
    std::vector<CcuRep::CcuBuf> buffers(config.lanes*msStride);
    std::vector<CcuRep::CompletedEvent> events(config.lanes);
    std::vector<CcuRep::Executor> executors(config.lanes);
    // Contiguous blocks are required: hardware loop expansion shifts MS/event
    // resource IDs by lane. Individual noncontiguous allocations are unsafe.
    GROUP_CHECK(CreateBlockCcuBuf(static_cast<uint32_t>(buffers.size()),buffers.data()));
    GROUP_CHECK(CreateBlockCompletedEvent(static_cast<uint32_t>(events.size()),events.data()));
    GROUP_CHECK(CreateBlockExecutor(static_cast<uint32_t>(executors.size()),executors.data()));
    std::array<CcuRep::LocalAddr,2> bodySrc,bodyDst;
    std::array<CcuRep::Variable,2> bodyBytes;
    const std::array<std::string,2> labels={{"a5-self-gc-first","a5-self-gc-repeat"}};
    for(size_t i=0;i<2;++i) {
        bodySrc[i]=CreateLocalAddr(); bodyDst[i]=CreateLocalAddr(); bodyBytes[i]=CreateVariable();
        CcuRep::LoopBlock block(this,labels[i]);
        block(bodySrc[i],bodyDst[i],bodyBytes[i]);
        GROUP_CHECK(LocalCopyNb(buffers[i*msStride],bodySrc[i],bodyBytes[i],events[i]));
        GROUP_CHECK(WaitEvent(events[i])); // MS must contain input before writing HBM
        GROUP_CHECK(LocalCopyNb(bodyDst[i],buffers[i*msStride],bodyBytes[i],events[i]));
        GROUP_CHECK(WaitEvent(events[i])); // no MS reuse before write completes
    }
    auto set=[&](size_t i,const CcuRep::Variable &len) {
        bodySrc[i].addr=src.addr; bodySrc[i].token=src.token;
        bodyDst[i].addr=dst.addr; bodyDst[i].token=dst.token; bodyBytes[i]=len;
    };
    auto slice=CreateVariable(), offset=CreateVariable();
    slice=config.sliceBytes; offset=GroupOffsetParam(config.sliceBytes);
    CCU_IF(go[0]!=0) {
        set(0,slice);
        auto call=Loop(labels[0]); call(bodySrc[0],bodyDst[0],bodyBytes[0]);
        auto cfg=CreateVariable(), parallel=CreateVariable();
        cfg=GroupLoopParam(config.BatchBytes(),0); cfg+=go[1];
        parallel=GroupParallelParam(config.lanes-1,0,1);
        CcuRep::LoopGroupCall(this).Run({call},{cfg},executors,parallel,offset);
    }
    CCU_IF(go[2]!=0) {
        src.addr+=go[0]; dst.addr+=go[0];
        set(0,go[3]); // residual first, as native CalGoSize/GroupCopy
        src.addr+=go[3]; dst.addr+=go[3];
        set(1,slice);
        auto first=Loop(labels[0]), repeat=Loop(labels[1]);
        first(bodySrc[0],bodyDst[0],bodyBytes[0]); repeat(bodySrc[1],bodyDst[1],bodyBytes[1]);
        auto cfg0=CreateVariable(), cfg1=CreateVariable();
        cfg0=GroupLoopParam(0,1); cfg1=GroupLoopParam(0,1);
        CcuRep::LoopGroupCall(this).Run({first,repeat},{cfg0,cfg1},executors,go[2],offset);
    }
    return HCCL_SUCCESS;
}
#undef GROUP_CHECK
} // namespace a5_ccu_urma_probe
