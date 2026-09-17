#include "one_way_write_kernel.h"

namespace a5_ccu_urma_probe {
namespace {
constexpr uint32_t OUTPUT_VAR_INDEX = 0;
constexpr uint32_t TOKEN_VAR_INDEX = 1;
constexpr uint32_t OUTPUT_NOTIFY_INDEX = 0;
constexpr uint32_t TOKEN_NOTIFY_INDEX = 1;
constexpr uint32_t COMPLETION_NOTIFY_INDEX = 2;
constexpr uint32_t NOTIFY_MASK = 1;
#define CCU_KERNEL_CHECK(expression) do { HcclResult status = (expression); \
    if (status != HCCL_SUCCESS) return status; } while (0)
} // namespace

hcomm::CcuKernelSignature OneWayWriteKernelArg::GetKernelSignature() const
{
    hcomm::CcuKernelSignature signature;
    signature.Append("A5CcuUrmaOneWayWriteV1");
    signature.Append(static_cast<uint32_t>(source_));
    return signature;
}

OneWayWriteKernel::OneWayWriteKernel(const hcomm::CcuKernelArg &arg) : hcomm::CcuKernel(arg)
{
    const auto *kernelArg = dynamic_cast<const OneWayWriteKernelArg *>(&arg);
    if (kernelArg != nullptr) source_ = kernelArg->IsSource();
}

HcclResult OneWayWriteKernel::Algorithm()
{
    if (channels_.size() != 1) return HCCL_E_PARA;
    const ChannelHandle channel = channels_.front();
    hcomm::CcuRep::Variable localInput = CreateVariable();
    hcomm::CcuRep::Variable localOutput = CreateVariable();
    hcomm::CcuRep::Variable localInputToken = CreateVariable();
    hcomm::CcuRep::Variable localOutputToken = CreateVariable();
    hcomm::CcuRep::Variable bytes = CreateVariable();
    Load(localInput); Load(localOutput); Load(localInputToken); Load(localOutputToken); Load(bytes);

    if (source_) {
        hcomm::CcuRep::Variable remoteOutput;
        hcomm::CcuRep::Variable remoteToken;
        CCU_KERNEL_CHECK(CreateVariable(channel, OUTPUT_VAR_INDEX, &remoteOutput));
        CCU_KERNEL_CHECK(CreateVariable(channel, TOKEN_VAR_INDEX, &remoteToken));
        CCU_KERNEL_CHECK(NotifyWait(channel, OUTPUT_NOTIFY_INDEX, NOTIFY_MASK));
        CCU_KERNEL_CHECK(NotifyWait(channel, TOKEN_NOTIFY_INDEX, NOTIFY_MASK));
        hcomm::CcuRep::LocalAddr source = CreateLocalAddr();
        source.addr = localInput; source.token = localInputToken;
        hcomm::CcuRep::RemoteAddr destination = CreateRemoteAddr();
        destination.addr = remoteOutput; destination.token = remoteToken;
        hcomm::CcuRep::CompletedEvent event = CreateCompletedEvent();
        CCU_KERNEL_CHECK(WriteNb(channel, destination, source, bytes, event));
        CCU_KERNEL_CHECK(WaitEvent(event));
        CCU_KERNEL_CHECK(NotifyRecord(channel, COMPLETION_NOTIFY_INDEX, NOTIFY_MASK));
    } else {
        CCU_KERNEL_CHECK(NotifyRecord(channel, OUTPUT_NOTIFY_INDEX, OUTPUT_VAR_INDEX,
                                      localOutput, NOTIFY_MASK));
        CCU_KERNEL_CHECK(NotifyRecord(channel, TOKEN_NOTIFY_INDEX, TOKEN_VAR_INDEX,
                                      localOutputToken, NOTIFY_MASK));
        CCU_KERNEL_CHECK(NotifyWait(channel, COMPLETION_NOTIFY_INDEX, NOTIFY_MASK));
    }
    return HCCL_SUCCESS;
}

std::vector<uint64_t> OneWayWriteKernel::GeneArgs(const hcomm::CcuTaskArg &arg)
{
    const auto *taskArg = dynamic_cast<const OneWayWriteTaskArg *>(&arg);
    if (taskArg == nullptr) return {};
    return {taskArg->inputAddr, taskArg->outputAddr, taskArg->inputToken,
            taskArg->outputToken, taskArg->bytes};
}

std::unique_ptr<hcomm::CcuKernel> CreateOneWayWriteKernel(const hcomm::CcuKernelArg &arg)
{
    return std::unique_ptr<hcomm::CcuKernel>(new OneWayWriteKernel(arg));
}
} // namespace a5_ccu_urma_probe
