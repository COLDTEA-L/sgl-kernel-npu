#ifndef A5_CCU_URMA_ONE_WAY_WRITE_KERNEL_H
#define A5_CCU_URMA_ONE_WAY_WRITE_KERNEL_H

#include <hcomm/ccu/ccu_kernel.h>

#include <cstdint>
#include <memory>

namespace a5_ccu_urma_probe {

class OneWayWriteKernelArg : public hcomm::CcuKernelArg {
public:
    OneWayWriteKernelArg(ChannelHandle channel, bool source) : source_(source)
    {
        channels.push_back(channel);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override;
    bool IsSource() const { return source_; }

private:
    bool source_;
};

class OneWayWriteTaskArg : public hcomm::CcuTaskArg {
public:
    OneWayWriteTaskArg(uint64_t inputAddr, uint64_t outputAddr,
                       uint64_t inputToken, uint64_t outputToken, uint64_t bytes)
        : inputAddr(inputAddr), outputAddr(outputAddr), inputToken(inputToken),
          outputToken(outputToken), bytes(bytes) {}
    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t inputToken;
    uint64_t outputToken;
    uint64_t bytes;
};

class OneWayWriteKernel : public hcomm::CcuKernel {
public:
    explicit OneWayWriteKernel(const hcomm::CcuKernelArg &arg);

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    bool source_ = false;
};

std::unique_ptr<hcomm::CcuKernel> CreateOneWayWriteKernel(const hcomm::CcuKernelArg &arg);

} // namespace a5_ccu_urma_probe
#endif
