#include "route_kernel.h"

namespace a5_ccu_urma_probe {
namespace {
constexpr uint32_t OUTPUT_VAR_INDEX = 0;
constexpr uint32_t TOKEN_VAR_INDEX = 1;
constexpr uint32_t OUTPUT_NOTIFY_INDEX = 0;
constexpr uint32_t TOKEN_NOTIFY_INDEX = 1;
constexpr uint32_t COMPLETION_NOTIFY_INDEX = 2;
constexpr uint32_t NOTIFY_MASK = 1;

#define CCU_KERNEL_CHECK(expression)          \
    do {                                      \
        HcclResult status = (expression);     \
        if (status != HCCL_SUCCESS) {         \
            return status;                    \
        }                                     \
    } while (0)
} // namespace

RouteKernelArg::RouteKernelArg(ChannelHandle channel, uint32_t routeIndex)
    : routeIndex_(routeIndex)
{
    channels.push_back(channel);
}

hcomm::CcuKernelSignature RouteKernelArg::GetKernelSignature() const
{
    hcomm::CcuKernelSignature signature;
    signature.Append("A5CcuUrmaRouteProbeV2");
    signature.Append(routeIndex_);
    return signature;
}

HcclResult RouteKernel::Algorithm()
{
    if (channels_.size() != 1) {
        return HCCL_E_PARA;
    }
    const ChannelHandle channel = channels_[0];

    hcomm::CcuRep::Variable localInput = CreateVariable();
    hcomm::CcuRep::Variable localOutput = CreateVariable();
    hcomm::CcuRep::Variable localInputToken = CreateVariable();
    hcomm::CcuRep::Variable localOutputToken = CreateVariable();
    hcomm::CcuRep::Variable bytes = CreateVariable();
    hcomm::CcuRep::Variable remoteOffset = CreateVariable();
    hcomm::CcuRep::Variable remoteOutput;
    hcomm::CcuRep::Variable remoteOutputToken;

    CCU_KERNEL_CHECK(CreateVariable(channel, OUTPUT_VAR_INDEX, &remoteOutput));
    CCU_KERNEL_CHECK(CreateVariable(channel, TOKEN_VAR_INDEX, &remoteOutputToken));

    Load(localInput);
    Load(localOutput);
    Load(localInputToken);
    Load(localOutputToken);
    Load(bytes);
    Load(remoteOffset);

    CCU_KERNEL_CHECK(NotifyRecord(channel, OUTPUT_NOTIFY_INDEX, OUTPUT_VAR_INDEX,
                                  localOutput, NOTIFY_MASK));
    CCU_KERNEL_CHECK(NotifyRecord(channel, TOKEN_NOTIFY_INDEX, TOKEN_VAR_INDEX,
                                  localOutputToken, NOTIFY_MASK));
    CCU_KERNEL_CHECK(NotifyWait(channel, OUTPUT_NOTIFY_INDEX, NOTIFY_MASK));
    CCU_KERNEL_CHECK(NotifyWait(channel, TOKEN_NOTIFY_INDEX, NOTIFY_MASK));

    hcomm::CcuRep::LocalAddr source = CreateLocalAddr();
    source.addr = localInput;
    source.token = localInputToken;
    hcomm::CcuRep::RemoteAddr destination = CreateRemoteAddr();
    destination.addr = remoteOutput;
    destination.addr += remoteOffset;
    destination.token = remoteOutputToken;

    hcomm::CcuRep::CompletedEvent event = CreateCompletedEvent();
    CCU_KERNEL_CHECK(WriteNb(channel, destination, source, bytes, event));
    CCU_KERNEL_CHECK(WaitEvent(event));

    CCU_KERNEL_CHECK(NotifyRecord(channel, COMPLETION_NOTIFY_INDEX, NOTIFY_MASK));
    CCU_KERNEL_CHECK(NotifyWait(channel, COMPLETION_NOTIFY_INDEX, NOTIFY_MASK));
    return HCCL_SUCCESS;
}

std::vector<uint64_t> RouteKernel::GeneArgs(const hcomm::CcuTaskArg &arg)
{
    const auto *taskArg = dynamic_cast<const RouteTaskArg *>(&arg);
    if (taskArg == nullptr) {
        return {};
    }
    return {taskArg->inputAddr, taskArg->outputAddr, taskArg->inputToken,
            taskArg->outputToken, taskArg->bytes, taskArg->remoteOffset};
}

std::unique_ptr<hcomm::CcuKernel> CreateRouteKernel(const hcomm::CcuKernelArg &arg)
{
    return std::unique_ptr<hcomm::CcuKernel>(new RouteKernel(arg));
}

} // namespace a5_ccu_urma_probe
