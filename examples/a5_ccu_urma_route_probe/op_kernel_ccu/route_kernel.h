#ifndef A5_CCU_URMA_ROUTE_PROBE_KERNEL_H
#define A5_CCU_URMA_ROUTE_PROBE_KERNEL_H

#include "grouped_local_copy_kernel.h"
#include "../common/channel_sync_policy.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace a5_ccu_urma_probe {

class RouteKernelArg : public hcomm::CcuKernelArg {
public:
    RouteKernelArg(const std::vector<ChannelHandle> &channels,
                   const std::vector<uint32_t> &routeIndices, bool localCopy = false, bool groupCopy = true,
                   ChannelSyncMode syncMode = ChannelSyncMode::BASELINE);
    hcomm::CcuKernelSignature GetKernelSignature() const override;
    bool localCopy() const { return localCopy_; }
    bool groupCopy() const { return localCopy_ && groupCopy_; }
    ChannelSyncMode syncMode() const { return syncMode_; }

private:
    std::vector<uint32_t> routeIndices_;
    bool localCopy_ = false;
    bool groupCopy_ = true;
    ChannelSyncMode syncMode_ = ChannelSyncMode::BASELINE;
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

class RouteKernel : public GroupedLocalCopyKernel {
public:
    explicit RouteKernel(const hcomm::CcuKernelArg &arg)
        : GroupedLocalCopyKernel(arg), localCopy_(dynamic_cast<const RouteKernelArg &>(arg).localCopy()),
          groupCopy_(dynamic_cast<const RouteKernelArg &>(arg).groupCopy()),
          syncMode_(dynamic_cast<const RouteKernelArg &>(arg).syncMode()) {}

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;
private:
    bool localCopy_ = false;
    bool groupCopy_ = false;
    ChannelSyncMode syncMode_ = ChannelSyncMode::BASELINE;
};

std::unique_ptr<hcomm::CcuKernel> CreateRouteKernel(const hcomm::CcuKernelArg &arg);

} // namespace a5_ccu_urma_probe

#endif // A5_CCU_URMA_ROUTE_PROBE_KERNEL_H
