#include "all_to_all_multiroute_kernel.h"

namespace a5_ccu_urma_probe {
namespace {
constexpr uint32_t OUTPUT_VAR_INDEX = 0;
constexpr uint32_t TOKEN_VAR_INDEX = 1;
constexpr uint32_t OUTPUT_NOTIFY_INDEX = 0;
constexpr uint32_t TOKEN_NOTIFY_INDEX = 1;
constexpr uint32_t COMPLETION_NOTIFY_INDEX = 2;
constexpr uint32_t NOTIFY_MASK = 1;
constexpr size_t MAX_EXPLICIT_PATHS = 64U;

#define CCU_KERNEL_CHECK(expression) do { \
    HcclResult status = (expression); \
    if (status != HCCL_SUCCESS) { return status; } \
} while (0)
} // namespace

AllToAllMultiRouteKernelArg::AllToAllMultiRouteKernelArg(
    const std::vector<ChannelHandle> &inputChannels,
    const std::vector<uint32_t> &routeIndices,
    bool serialized) : routeIndices_(routeIndices), serialized_(serialized)
{
    channels = inputChannels;
}

hcomm::CcuKernelSignature AllToAllMultiRouteKernelArg::GetKernelSignature() const
{
    hcomm::CcuKernelSignature signature;
    signature.Append("A5CcuUrmaMultiRouteAllToAllV3");
    signature.Append(static_cast<uint32_t>(serialized_ ? 1U : 0U));
    for (const uint32_t routeIndex : routeIndices_) {
        signature.Append(routeIndex);
    }
    return signature;
}

AllToAllMultiRouteKernel::AllToAllMultiRouteKernel(const hcomm::CcuKernelArg &arg)
    : hcomm::CcuKernel(arg)
{
    const auto *typedArg = dynamic_cast<const AllToAllMultiRouteKernelArg *>(&arg);
    serialized_ = typedArg != nullptr && typedArg->IsSerialized();
}

HcclResult AllToAllMultiRouteKernel::Algorithm()
{
    if (channels_.empty() || channels_.size() > MAX_EXPLICIT_PATHS) {
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
    sourceOffsets.reserve(channels_.size());
    remoteOffsets.reserve(channels_.size());
    pathBytes.reserve(channels_.size());
    remoteOutputs.reserve(channels_.size());
    remoteTokens.reserve(channels_.size());
    for (size_t i = 0; i < channels_.size(); ++i) {
        sourceOffsets.push_back(CreateVariable());
        remoteOffsets.push_back(CreateVariable());
        pathBytes.push_back(CreateVariable());
        Load(sourceOffsets.back()); Load(remoteOffsets.back()); Load(pathBytes.back());
        remoteOutputs.emplace_back(); remoteTokens.emplace_back();
        CCU_KERNEL_CHECK(CreateVariable(channels_[i], OUTPUT_VAR_INDEX, &remoteOutputs.back()));
        CCU_KERNEL_CHECK(CreateVariable(channels_[i], TOKEN_VAR_INDEX, &remoteTokens.back()));
        CCU_KERNEL_CHECK(NotifyRecord(channels_[i], OUTPUT_NOTIFY_INDEX, OUTPUT_VAR_INDEX,
                                      output, NOTIFY_MASK));
        CCU_KERNEL_CHECK(NotifyRecord(channels_[i], TOKEN_NOTIFY_INDEX, TOKEN_VAR_INDEX,
                                      outputToken, NOTIFY_MASK));
    }
    for (const ChannelHandle channel : channels_) {
        CCU_KERNEL_CHECK(NotifyWait(channel, OUTPUT_NOTIFY_INDEX, NOTIFY_MASK));
        CCU_KERNEL_CHECK(NotifyWait(channel, TOKEN_NOTIFY_INDEX, NOTIFY_MASK));
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
        // The serial kernel is a controlled experiment: it differs from the
        // production kernel only by waiting after each route submission.
        if (serialized_) {
            CCU_KERNEL_CHECK(WaitEvent(remoteEvents.back()));
        }
    }

    CCU_KERNEL_CHECK(WaitEvent(localEvent));
    if (!serialized_) {
        for (auto &event : remoteEvents) {
            CCU_KERNEL_CHECK(WaitEvent(event));
        }
    }

    // A local WriteNb completion is not a substitute for the remote channel's
    // post-sync.  Every route owns an independent channel/jetty and therefore
    // every route must participate in the completion handshake.  Synchronizing
    // only channels_.front() lets a later invocation reuse output/event state
    // while writes issued on the other channels are still becoming visible.
    // This mirrors the native HCCL AllToAll/MultiJetty post-sync protocol.
    for (const ChannelHandle channel : channels_) {
        CCU_KERNEL_CHECK(NotifyRecord(channel, COMPLETION_NOTIFY_INDEX, NOTIFY_MASK));
    }
    for (const ChannelHandle channel : channels_) {
        CCU_KERNEL_CHECK(NotifyWait(channel, COMPLETION_NOTIFY_INDEX, NOTIFY_MASK));
    }
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
