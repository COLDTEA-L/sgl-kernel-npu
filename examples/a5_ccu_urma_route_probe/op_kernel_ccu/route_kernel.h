#ifndef A5_CCU_URMA_ROUTE_PROBE_KERNEL_H
#define A5_CCU_URMA_ROUTE_PROBE_KERNEL_H

#include <hcomm/ccu/ccu_kernel.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace a5_ccu_urma_probe {

class RouteKernelArg : public hcomm::CcuKernelArg {
public:
    RouteKernelArg(ChannelHandle channel, uint32_t routeIndex);
    hcomm::CcuKernelSignature GetKernelSignature() const override;

private:
    uint32_t routeIndex_;
};

class RouteTaskArg : public hcomm::CcuTaskArg {
public:
    RouteTaskArg(uint64_t inputAddr, uint64_t outputAddr, uint64_t inputToken,
                 uint64_t outputToken, uint64_t bytes, uint64_t remoteOffset)
        : inputAddr(inputAddr), outputAddr(outputAddr), inputToken(inputToken),
          outputToken(outputToken), bytes(bytes), remoteOffset(remoteOffset) {}

    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t inputToken;
    uint64_t outputToken;
    uint64_t bytes;
    uint64_t remoteOffset;
};

class RouteKernel : public hcomm::CcuKernel {
public:
    explicit RouteKernel(const hcomm::CcuKernelArg &arg) : hcomm::CcuKernel(arg) {}

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;
};

std::unique_ptr<hcomm::CcuKernel> CreateRouteKernel(const hcomm::CcuKernelArg &arg);

} // namespace a5_ccu_urma_probe

#endif // A5_CCU_URMA_ROUTE_PROBE_KERNEL_H
