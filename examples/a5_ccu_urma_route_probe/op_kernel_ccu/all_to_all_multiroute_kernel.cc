#include "all_to_all_multiroute_kernel.h"

namespace a5_ccu_urma_probe {
namespace {
constexpr uint32_t OUTPUT_VAR_INDEX = 0;
constexpr uint32_t TOKEN_VAR_INDEX = 1;
constexpr uint32_t PRE_SYNC_NOTIFY_INDEX = 0;
constexpr uint32_t POST_SYNC_NOTIFY_INDEX = 1;
constexpr uint32_t OUTPUT_MASK = 1;
constexpr uint32_t TOKEN_MASK = 2;
constexpr uint32_t PRE_SYNC_MASK = OUTPUT_MASK | TOKEN_MASK;
constexpr uint32_t POST_SYNC_MASK = 1;

#define CCU_KERNEL_CHECK(expression) do { \
    HcclResult status = (expression); \
    if (status != HCCL_SUCCESS) { return status; } \
} while (0)
} // namespace

AllToAllMultiRouteKernelArg::AllToAllMultiRouteKernelArg(
    const std::vector<ChannelHandle> &inputChannels,
    const std::vector<uint32_t> &routeIndices) : routeIndices_(routeIndices)
{
    channels = inputChannels;
}

hcomm::CcuKernelSignature AllToAllMultiRouteKernelArg::GetKernelSignature() const
{
    hcomm::CcuKernelSignature signature;
    signature.Append("A5CcuUrmaMultiRouteAllToAllV1");
    for (const uint32_t routeIndex : routeIndices_) {
        signature.Append(routeIndex);
    }
    return signature;
}

HcclResult AllToAllMultiRouteKernel::Algorithm()
{
    if (channels_.empty() || channels_.size() > 8) {
        return HCCL_E_PARA;
    }

    hcomm::CcuRep::Variable input = CreateVariable();
    hcomm::CcuRep::Variable output = CreateVariable();
    hcomm::CcuRep::Variable inputToken = CreateVariable();
    hcomm::CcuRep::Variable outputToken = CreateVariable();
    hcomm::CcuRep::Variable selfSourceOffset = CreateVariable();
    hcomm::CcuRep::Variable selfDestinationOffset = CreateVariable();
    hcomm::CcuRep::Variable selfBytes = CreateVariable();
    Load(input); Load(output); Load(inputToken); Load(outputToken);
    Load(selfSourceOffset); Load(selfDestinationOffset); Load(selfBytes);

    std::vector<hcomm::CcuRep::Variable> sourceOffsets;
    std::vector<hcomm::CcuRep::Variable> remoteOffsets;
    std::vector<hcomm::CcuRep::Variable> pathBytes;
    std::vector<hcomm::CcuRep::Variable> remoteOutputs;
    std::vector<hcomm::CcuRep::Variable> remoteTokens;
    for (size_t i = 0; i < channels_.size(); ++i) {
        sourceOffsets.push_back(CreateVariable());
        remoteOffsets.push_back(CreateVariable());
        pathBytes.push_back(CreateVariable());
        Load(sourceOffsets.back()); Load(remoteOffsets.back()); Load(pathBytes.back());
        remoteOutputs.emplace_back(); remoteTokens.emplace_back();
        CCU_KERNEL_CHECK(CreateVariable(channels_[i], OUTPUT_VAR_INDEX, &remoteOutputs.back()));
        CCU_KERNEL_CHECK(CreateVariable(channels_[i], TOKEN_VAR_INDEX, &remoteTokens.back()));
        CCU_KERNEL_CHECK(NotifyRecord(channels_[i], PRE_SYNC_NOTIFY_INDEX, OUTPUT_VAR_INDEX,
                                      output, OUTPUT_MASK));
        CCU_KERNEL_CHECK(NotifyRecord(channels_[i], PRE_SYNC_NOTIFY_INDEX, TOKEN_VAR_INDEX,
                                      outputToken, TOKEN_MASK));
    }
    for (const ChannelHandle channel : channels_) {
        CCU_KERNEL_CHECK(NotifyWait(channel, PRE_SYNC_NOTIFY_INDEX, PRE_SYNC_MASK));
    }

    hcomm::CcuRep::LocalAddr localSource = CreateLocalAddr();
    localSource.addr = input; localSource.addr += selfSourceOffset; localSource.token = inputToken;
    hcomm::CcuRep::LocalAddr localDestination = CreateLocalAddr();
    localDestination.addr = output; localDestination.addr += selfDestinationOffset;
    localDestination.token = outputToken;
    hcomm::CcuRep::CompletedEvent localEvent = CreateCompletedEvent();
    CCU_KERNEL_CHECK(LocalCopyNb(localDestination, localSource, selfBytes, localEvent));

    std::vector<hcomm::CcuRep::CompletedEvent> remoteEvents;
    for (size_t i = 0; i < channels_.size(); ++i) {
        hcomm::CcuRep::LocalAddr source = CreateLocalAddr();
        source.addr = input; source.addr += sourceOffsets[i]; source.token = inputToken;
        hcomm::CcuRep::RemoteAddr destination = CreateRemoteAddr();
        destination.addr = remoteOutputs[i]; destination.addr += remoteOffsets[i];
        destination.token = remoteTokens[i];
        remoteEvents.push_back(CreateCompletedEvent());
        CCU_KERNEL_CHECK(WriteNb(channels_[i], destination, source, pathBytes[i], remoteEvents.back()));
    }

    CCU_KERNEL_CHECK(WaitEvent(localEvent));
    for (auto &event : remoteEvents) {
        CCU_KERNEL_CHECK(WaitEvent(event));
    }

    // All data routes target the same peer. One peer-level completion handshake
    // is sufficient after every local completion event has fired.
    CCU_KERNEL_CHECK(NotifyRecord(channels_.front(), POST_SYNC_NOTIFY_INDEX, POST_SYNC_MASK));
    CCU_KERNEL_CHECK(NotifyWait(channels_.front(), POST_SYNC_NOTIFY_INDEX, POST_SYNC_MASK));
    return HCCL_SUCCESS;
}

std::vector<uint64_t> AllToAllMultiRouteKernel::GeneArgs(const hcomm::CcuTaskArg &arg)
{
    const auto *taskArg = dynamic_cast<const AllToAllMultiRouteTaskArg *>(&arg);
    if (taskArg == nullptr) {
        return {};
    }
    std::vector<uint64_t> args = {
        taskArg->inputAddr, taskArg->outputAddr, taskArg->inputToken, taskArg->outputToken,
        taskArg->selfSourceOffset, taskArg->selfDestinationOffset, taskArg->selfBytes
    };
    for (size_t i = 0; i < taskArg->pathBytes.size(); ++i) {
        args.push_back(taskArg->sourceOffsets[i]);
        args.push_back(taskArg->remoteOffsets[i]);
        args.push_back(taskArg->pathBytes[i]);
    }
    return args;
}

std::unique_ptr<hcomm::CcuKernel> CreateAllToAllMultiRouteKernel(const hcomm::CcuKernelArg &arg)
{
    return std::unique_ptr<hcomm::CcuKernel>(new AllToAllMultiRouteKernel(arg));
}

} // namespace a5_ccu_urma_probe
