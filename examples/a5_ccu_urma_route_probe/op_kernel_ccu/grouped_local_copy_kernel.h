#ifndef A5_GROUPED_LOCAL_COPY_KERNEL_H
#define A5_GROUPED_LOCAL_COPY_KERNEL_H
#include "public_ccu.h"
#include "../common/group_copy_plan.h"
namespace a5_ccu_urma_probe {
class GroupedLocalCopyKernel : public a5ccu::CcuKernel {
public:
    explicit GroupedLocalCopyKernel(const a5ccu::CcuKernelArg &arg) : a5ccu::CcuKernel(arg) {}
protected:
    // Synchronous group join in the CCU graph, NOT a CPU synchronization.
    // Issue remote WriteNb BEFORE this, like native HCCL DoAllToAll.
    HcclResult GroupedLocalCopy(a5ccu::CcuRep::LocalAddr destination,
        a5ccu::CcuRep::LocalAddr source, std::array<a5ccu::CcuRep::Variable,4> go,
        const GroupCopyConfig &config = GroupCopyConfig());
};
} // namespace a5_ccu_urma_probe
#endif
