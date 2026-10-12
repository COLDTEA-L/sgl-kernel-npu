#include "route_kernel.h"
#include "../common/path_layout.h"

namespace a5_ccu_urma_probe {
namespace {
constexpr uint32_t OUTPUT_VAR_INDEX = 0;
constexpr uint32_t TOKEN_VAR_INDEX = 1;
constexpr uint32_t OUTPUT_NOTIFY_INDEX = 0;
constexpr uint32_t TOKEN_NOTIFY_INDEX = 1;
constexpr uint32_t COMPLETION_NOTIFY_INDEX = 2;
constexpr uint32_t NOTIFY_MASK = 1;
constexpr size_t MAX_ROUTE_PATHS = 64U;

#define CCU_KERNEL_CHECK(expression)          \
    do {                                      \
        HcclResult status = (expression);     \
        if (status != HCCL_SUCCESS) {         \
            return status;                    \
        }                                     \
    } while (0)
} // namespace

RouteKernelArg::RouteKernelArg(const std::vector<ChannelHandle> &inputChannels,
                               const std::vector<uint32_t> &routeIndices, bool localCopy)
    : routeIndices_(routeIndices), localCopy_(localCopy)
{
    channels = inputChannels;
}

hcomm::CcuKernelSignature RouteKernelArg::GetKernelSignature() const
{
    hcomm::CcuKernelSignature signature;
    signature.Append(MakePathKernelSignature(localCopy_ ? "A5CcuUrmaLocalAndRemoteV1" :
        "A5CcuUrmaMultiRouteV2", routeIndices_, channels));
    return signature;
}

HcclResult RouteKernel::Algorithm()
{
    if (channels_.empty() || channels_.size() > MAX_ROUTE_PATHS ||
        (localCopy_ ? 6U : 4U) + 3U * channels_.size() > 48U) {
        return HCCL_E_PARA;
    }

    hcomm::CcuRep::Variable localInput = CreateVariable();
    hcomm::CcuRep::Variable localOutput = CreateVariable();
    hcomm::CcuRep::Variable localInputToken = CreateVariable();
    hcomm::CcuRep::Variable localOutputToken = CreateVariable();
    Load(localInput);
    Load(localOutput);
    Load(localInputToken);
    Load(localOutputToken);

    hcomm::CcuRep::Variable selfOffset;
    hcomm::CcuRep::Variable selfBytes;
    if (localCopy_) {
        selfOffset = CreateVariable();
        selfBytes = CreateVariable();
        Load(selfOffset);
        Load(selfBytes);
    }

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
        Load(sourceOffsets.back());
        Load(remoteOffsets.back());
        Load(pathBytes.back());
        remoteOutputs.emplace_back();
        remoteTokens.emplace_back();
        CCU_KERNEL_CHECK(CreateVariable(channels_[i], OUTPUT_VAR_INDEX, &remoteOutputs.back()));
        CCU_KERNEL_CHECK(CreateVariable(channels_[i], TOKEN_VAR_INDEX, &remoteTokens.back()));
        CCU_KERNEL_CHECK(NotifyRecord(channels_[i], OUTPUT_NOTIFY_INDEX, OUTPUT_VAR_INDEX,
                                      localOutput, NOTIFY_MASK));
        CCU_KERNEL_CHECK(NotifyRecord(channels_[i], TOKEN_NOTIFY_INDEX, TOKEN_VAR_INDEX,
                                      localOutputToken, NOTIFY_MASK));
    }
    // Separate local completion resource; not a Channel notify slot. Submit
    // before peer waits, but DO NOT wait for it before submitting WriteNb.
    // The host validated self row vs incoming peer rows are disjoint.
    hcomm::CcuRep::CompletedEvent localEvent;
    if (localCopy_) {
        localEvent = CreateCompletedEvent();
        hcomm::CcuRep::LocalAddr src = CreateLocalAddr();
        hcomm::CcuRep::LocalAddr dst = CreateLocalAddr();
        src.addr = localInput; src.addr += selfOffset; src.token = localInputToken;
        dst.addr = localOutput; dst.addr += selfOffset; dst.token = localOutputToken;
        CCU_KERNEL_CHECK(LocalCopyNb(dst, src, selfBytes, localEvent));
    }
    for (size_t i = 0; i < channels_.size(); ++i) {
        CCU_KERNEL_CHECK(NotifyWait(channels_[i], OUTPUT_NOTIFY_INDEX, NOTIFY_MASK));
        CCU_KERNEL_CHECK(NotifyWait(channels_[i], TOKEN_NOTIFY_INDEX, NOTIFY_MASK));
    }

    std::vector<hcomm::CcuRep::CompletedEvent> events;
    events.reserve(channels_.size());
    for (size_t i = 0; i < channels_.size(); ++i) {
        hcomm::CcuRep::LocalAddr source = CreateLocalAddr();
        source.addr = localInput;
        source.addr += sourceOffsets[i];
        source.token = localInputToken;
        hcomm::CcuRep::RemoteAddr destination = CreateRemoteAddr();
        destination.addr = remoteOutputs[i];
        destination.addr += remoteOffsets[i];
        destination.token = remoteTokens[i];
        events.push_back(CreateCompletedEvent());
        CCU_KERNEL_CHECK(WriteNb(channels_[i], destination, source, pathBytes[i], events.back()));
    }
    // Every remote WriteNb has now been issued. Local and remote work may
    // overlap; waiting here does not serialize their submission.
    if (localCopy_) CCU_KERNEL_CHECK(WaitEvent(localEvent));
    for (auto &event : events) {
        CCU_KERNEL_CHECK(WaitEvent(event));
    }
    for (const ChannelHandle channel : channels_) {
        CCU_KERNEL_CHECK(NotifyRecord(channel, COMPLETION_NOTIFY_INDEX, NOTIFY_MASK));
    }
    for (const ChannelHandle channel : channels_) {
        CCU_KERNEL_CHECK(NotifyWait(channel, COMPLETION_NOTIFY_INDEX, NOTIFY_MASK));
    }
    return HCCL_SUCCESS;
}

std::vector<uint64_t> RouteKernel::GeneArgs(const hcomm::CcuTaskArg &arg)
{
    const auto *taskArg = dynamic_cast<const RouteTaskArg *>(&arg);
    if (taskArg == nullptr || taskArg->pathBytes.size() != channels_.size() ||
        taskArg->sourceOffsets.size() != channels_.size() ||
        taskArg->remoteOffsets.size() != channels_.size() ||
        (localCopy_ && taskArg->selfBytes == 0) ||
        (localCopy_ ? 6 : 4) + 3 * channels_.size() > 48) {
        return {};
    }
    std::vector<uint64_t> args = {taskArg->inputAddr, taskArg->outputAddr,
                                  taskArg->inputToken, taskArg->outputToken};
    if (localCopy_) {
        args.push_back(taskArg->selfOffset);
        args.push_back(taskArg->selfBytes);
    }
    for (size_t i = 0; i < taskArg->pathBytes.size(); ++i) {
        args.push_back(taskArg->sourceOffsets[i]);
        args.push_back(taskArg->remoteOffsets[i]);
        args.push_back(taskArg->pathBytes[i]);
    }
    return args;
}

std::unique_ptr<hcomm::CcuKernel> CreateRouteKernel(const hcomm::CcuKernelArg &arg)
{
    return std::unique_ptr<hcomm::CcuKernel>(new RouteKernel(arg));
}

} // namespace a5_ccu_urma_probe
