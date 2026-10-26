#ifndef A5_CCU_ALLTOALLV_KERNEL_H
#define A5_CCU_ALLTOALLV_KERNEL_H
#include "grouped_local_copy_kernel.h"
#include "../common/alltoallv_layout.h"
#include <memory>
namespace a5_ccu_urma_probe {
class AllToAllVKernelArg : public hcomm::CcuKernelArg {
public:
    explicit AllToAllVKernelArg(const std::vector<ChannelHandle> &input) { channels = input; }
    hcomm::CcuKernelSignature GetKernelSignature() const override;
};
class AllToAllVTaskArg : public hcomm::CcuTaskArg {
public:
    uint64_t input, output, inputToken, outputToken;
    AllToAllVLayout layout;
    AllToAllVTaskArg(uint64_t in, uint64_t out, uint64_t it, uint64_t ot, const AllToAllVLayout &l)
        : input(in), output(out), inputToken(it), outputToken(ot), layout(l) {}
};
class AllToAllVKernel : public GroupedLocalCopyKernel {
public:
    explicit AllToAllVKernel(const hcomm::CcuKernelArg &arg) : GroupedLocalCopyKernel(arg) {}
protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;
};
std::unique_ptr<hcomm::CcuKernel> CreateAllToAllVKernel(const hcomm::CcuKernelArg &arg);
} // namespace a5_ccu_urma_probe
#endif
