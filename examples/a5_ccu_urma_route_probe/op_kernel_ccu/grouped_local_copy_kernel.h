#ifndef A5_GROUPED_LOCAL_COPY_KERNEL_H
#define A5_GROUPED_LOCAL_COPY_KERNEL_H
#include <hcomm/ccu/ccu_kernel.h>
#include "../common/group_copy_plan.h"
namespace a5_ccu_urma_probe {
class GroupedLocalCopyKernel : public hcomm::CcuKernel {
public:
    explicit GroupedLocalCopyKernel(const hcomm::CcuKernelArg &arg) : hcomm::CcuKernel(arg) {}
protected:
    // Synchronous group join in the CCU graph, NOT a CPU synchronization.
    // Issue remote WriteNb BEFORE this, like native HCCL DoAllToAll.
    HcclResult GroupedLocalCopy(hcomm::CcuRep::LocalAddr destination,
        hcomm::CcuRep::LocalAddr source, const std::array<hcomm::CcuRep::Variable,4> &go);
};
} // namespace a5_ccu_urma_probe
#endif
