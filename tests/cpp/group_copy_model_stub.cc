// CPU-only semantic stand-in for GroupedLocalCopy. Production links the SDK
// implementation instead. This model tests byte placement/call order, not MS
// allocation, executor translation, firmware parallelism or device timing.
#include "../../examples/a5_ccu_urma_route_probe/op_kernel_ccu/grouped_local_copy_kernel.h"
namespace a5_ccu_urma_probe {
HcclResult GroupedLocalCopyKernel::GroupedLocalCopy(hcomm::CcuRep::LocalAddr dst,
    hcomm::CcuRep::LocalAddr src,std::array<hcomm::CcuRep::Variable,4> go,
    const GroupCopyConfig &config)
{
    hcomm::Instruction i; i.op="group_local"; i.dst=dst.addr; i.src=src.addr;
    i.len.eval=[go,config](const hcomm::CcuRep::Env &e) {
        auto parallel=go[2].eval(e), residual=go[3].eval(e), bulk=go[0].eval(e);
        if(!parallel) return bulk;
        auto repeat=(parallel>>55)&0xff, total=(parallel>>41)&0x7f;
        return bulk+residual+(total==2 ? repeat+1 : repeat)*config.sliceBytes;
    };
    instructions.push_back(i);
    return HCCL_SUCCESS;
}
} // namespace a5_ccu_urma_probe
