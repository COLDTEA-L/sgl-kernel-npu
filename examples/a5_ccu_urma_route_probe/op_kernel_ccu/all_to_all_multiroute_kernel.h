#ifndef A5_CCU_URMA_ALL_TO_ALL_MULTI_ROUTE_KERNEL_H
#define A5_CCU_URMA_ALL_TO_ALL_MULTI_ROUTE_KERNEL_H

#include <hcomm/ccu/ccu_kernel.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace a5_ccu_urma_probe {

class AllToAllMultiRouteKernelArg : public hcomm::CcuKernelArg {
public:
    AllToAllMultiRouteKernelArg(const std::vector<ChannelHandle> &channels,
                                const std::vector<uint32_t> &routeIndices);
    hcomm::CcuKernelSignature GetKernelSignature() const override;

private:
    std::vector<uint32_t> routeIndices_;
};

class AllToAllMultiRouteTaskArg : public hcomm::CcuTaskArg {
public:
    AllToAllMultiRouteTaskArg(uint64_t inputAddr, uint64_t outputAddr,
                              uint64_t inputToken, uint64_t outputToken,
                              uint64_t selfSourceOffset, uint64_t selfDestinationOffset,
                              uint64_t selfBytes,
                              const std::vector<uint64_t> &sourceOffsets,
                              const std::vector<uint64_t> &remoteOffsets,
                              const std::vector<uint64_t> &pathBytes)
        : inputAddr(inputAddr), outputAddr(outputAddr), inputToken(inputToken),
          outputToken(outputToken), selfSourceOffset(selfSourceOffset),
          selfDestinationOffset(selfDestinationOffset), selfBytes(selfBytes),
          sourceOffsets(sourceOffsets), remoteOffsets(remoteOffsets), pathBytes(pathBytes) {}

    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t inputToken;
    uint64_t outputToken;
    uint64_t selfSourceOffset;
    uint64_t selfDestinationOffset;
    uint64_t selfBytes;
    std::vector<uint64_t> sourceOffsets;
    std::vector<uint64_t> remoteOffsets;
    std::vector<uint64_t> pathBytes;
};

class AllToAllMultiRouteKernel : public hcomm::CcuKernel {
public:
    explicit AllToAllMultiRouteKernel(const hcomm::CcuKernelArg &arg) : hcomm::CcuKernel(arg) {}

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;
};

std::unique_ptr<hcomm::CcuKernel> CreateAllToAllMultiRouteKernel(const hcomm::CcuKernelArg &arg);

} // namespace a5_ccu_urma_probe

#endif
