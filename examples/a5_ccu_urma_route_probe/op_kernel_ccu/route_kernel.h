#ifndef A5_CCU_URMA_ROUTE_PROBE_KERNEL_H
#define A5_CCU_URMA_ROUTE_PROBE_KERNEL_H

#include <hcomm/ccu/ccu_kernel.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace a5_ccu_urma_probe {

class RouteKernelArg : public hcomm::CcuKernelArg {
public:
    RouteKernelArg(const std::vector<ChannelHandle> &channels,
                   const std::vector<uint32_t> &routeIndices, bool localCopy = false);
    hcomm::CcuKernelSignature GetKernelSignature() const override;
    bool localCopy() const { return localCopy_; }

private:
    std::vector<uint32_t> routeIndices_;
    bool localCopy_ = false;
};

class RouteTaskArg : public hcomm::CcuTaskArg {
public:
    RouteTaskArg(uint64_t inputAddr, uint64_t outputAddr, uint64_t inputToken,
                 uint64_t outputToken, const std::vector<uint64_t> &sourceOffsets,
                 const std::vector<uint64_t> &remoteOffsets,
                 const std::vector<uint64_t> &pathBytes,
                 uint64_t selfOffset = 0, uint64_t selfBytes = 0)
        : inputAddr(inputAddr), outputAddr(outputAddr), inputToken(inputToken),
          outputToken(outputToken), sourceOffsets(sourceOffsets),
          remoteOffsets(remoteOffsets), pathBytes(pathBytes),
          selfOffset(selfOffset), selfBytes(selfBytes) {}

    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t inputToken;
    uint64_t outputToken;
    std::vector<uint64_t> sourceOffsets;
    std::vector<uint64_t> remoteOffsets;
    std::vector<uint64_t> pathBytes;
    uint64_t selfOffset;
    uint64_t selfBytes;
};

class RouteKernel : public hcomm::CcuKernel {
public:
    explicit RouteKernel(const hcomm::CcuKernelArg &arg)
        : hcomm::CcuKernel(arg), localCopy_(dynamic_cast<const RouteKernelArg &>(arg).localCopy()) {}

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;
private:
    bool localCopy_ = false;
};

std::unique_ptr<hcomm::CcuKernel> CreateRouteKernel(const hcomm::CcuKernelArg &arg);

} // namespace a5_ccu_urma_probe

#endif // A5_CCU_URMA_ROUTE_PROBE_KERNEL_H
