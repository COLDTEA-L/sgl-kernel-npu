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
                               const std::vector<uint32_t> &routeIndices, bool localCopy, bool groupCopy,
                               ChannelSyncMode syncMode)
    : routeIndices_(routeIndices), localCopy_(localCopy), groupCopy_(groupCopy), syncMode_(syncMode)
{
    channels = inputChannels;
}

a5ccu::CcuKernelSignature RouteKernelArg::GetKernelSignature() const
{
    a5ccu::CcuKernelSignature signature;
    signature.Append(MakePathKernelSignature(groupCopy() ? "A5CcuUrmaGroupedSelf8x32KV1" :
        localCopy_ ? "A5CcuUrmaLocalAndRemoteV1" :
        "A5CcuUrmaMultiRouteV2", routeIndices_, channels));
    signature.Append(std::string(":channel-sync-v1:") + ChannelSyncName(syncMode_));
    return signature;
}

HcclResult RouteKernel::Algorithm()
{
    if (channels_.empty() || channels_.size() > MAX_ROUTE_PATHS ||
        (groupCopy_ ? 9U : localCopy_ ? 6U : 4U) + 3U * channels_.size() > 48U) {
        return HCCL_E_PARA;
    }

    a5ccu::CcuRep::Variable localInput = CreateVariable();
    a5ccu::CcuRep::Variable localOutput = CreateVariable();
    a5ccu::CcuRep::Variable localInputToken = CreateVariable();
    a5ccu::CcuRep::Variable localOutputToken = CreateVariable();
    Load(localInput);
    Load(localOutput);
    Load(localInputToken);
    Load(localOutputToken);

    a5ccu::CcuRep::Variable selfOffset;
    a5ccu::CcuRep::Variable selfBytes;
    std::array<a5ccu::CcuRep::Variable,4> selfGo;
    if (localCopy_) {
        selfOffset = CreateVariable();
        Load(selfOffset);
        if(groupCopy_) {
            for(auto &v:selfGo) { v=CreateVariable(); Load(v); }
        } else { selfBytes=CreateVariable(); Load(selfBytes); }
    }

    std::vector<a5ccu::CcuRep::Variable> sourceOffsets;
    std::vector<a5ccu::CcuRep::Variable> remoteOffsets;
    std::vector<a5ccu::CcuRep::Variable> pathBytes;
    std::vector<a5ccu::CcuRep::Variable> remoteOutputs;
    std::vector<a5ccu::CcuRep::Variable> remoteTokens;
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
        // Distinct bits in the SAME notify slot; readiness requires BOTH.
        // Publish all Channels before any wait to avoid reciprocal wait cycles.
        const bool packed = syncMode_ != ChannelSyncMode::BASELINE;
        CCU_KERNEL_CHECK(NotifyRecord(channels_[i], packed ? OUTPUT_NOTIFY_INDEX : TOKEN_NOTIFY_INDEX,
                                      TOKEN_VAR_INDEX, localOutputToken, packed ? 2U : NOTIFY_MASK));
    }
    // Separate local completion resource; not a Channel notify slot. Submit
    // before peer waits, but DO NOT wait for it before submitting WriteNb.
    // The host validated self row vs incoming peer rows are disjoint.
    a5ccu::CcuRep::CompletedEvent localEvent;
    if (localCopy_ && !groupCopy_) {
        localEvent = CreateCompletedEvent();
        a5ccu::CcuRep::LocalAddr src = CreateLocalAddr();
        a5ccu::CcuRep::LocalAddr dst = CreateLocalAddr();
        src.addr = localInput; src.addr += selfOffset; src.token = localInputToken;
        dst.addr = localOutput; dst.addr += selfOffset; dst.token = localOutputToken;
        CCU_KERNEL_CHECK(LocalCopyNb(dst, src, selfBytes, localEvent));
    }
    auto waitMetadata = [&](size_t i) -> HcclResult {
        if (syncMode_ != ChannelSyncMode::BASELINE)
            return NotifyWait(channels_[i], OUTPUT_NOTIFY_INDEX, 3U);
        CCU_KERNEL_CHECK(NotifyWait(channels_[i], OUTPUT_NOTIFY_INDEX, NOTIFY_MASK));
        return NotifyWait(channels_[i], TOKEN_NOTIFY_INDEX, NOTIFY_MASK);
    };
    if (syncMode_ != ChannelSyncMode::PACKED_READY) {
        for (size_t i = 0; i < channels_.size(); ++i) CCU_KERNEL_CHECK(waitMetadata(i));
    }

    std::vector<a5ccu::CcuRep::CompletedEvent> events;
    events.reserve(channels_.size());
    for (size_t i = 0; i < channels_.size(); ++i) {
        // Ordered interpreter, NOT an out-of-order readiness scheduler.
        // Do not wait for write completion here: submit every path first.
        if (syncMode_ == ChannelSyncMode::PACKED_READY) CCU_KERNEL_CHECK(waitMetadata(i));
        a5ccu::CcuRep::LocalAddr source = CreateLocalAddr();
        source.addr = localInput;
        source.addr += sourceOffsets[i];
        source.token = localInputToken;
        a5ccu::CcuRep::RemoteAddr destination = CreateRemoteAddr();
        destination.addr = remoteOutputs[i];
        destination.addr += remoteOffsets[i];
        destination.token = remoteTokens[i];
        events.push_back(CreateCompletedEvent());
        CCU_KERNEL_CHECK(WriteNb(channels_[i], destination, source, pathBytes[i], events.back()));
    }
    // Native order: issue ALL remote writes first, then grouped local copy.
    // Group's internal MS read/write waits do not delay remote submission.
    if(groupCopy_) {
        auto src=CreateLocalAddr(),dst=CreateLocalAddr();
        src.addr=localInput; src.addr+=selfOffset; src.token=localInputToken;
        dst.addr=localOutput; dst.addr+=selfOffset; dst.token=localOutputToken;
        CCU_KERNEL_CHECK(GroupedLocalCopy(dst,src,selfGo));
    } else if (localCopy_) CCU_KERNEL_CHECK(WaitEvent(localEvent));
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

std::vector<uint64_t> RouteKernel::GeneArgs(const a5ccu::CcuTaskArg &arg)
{
    const auto *taskArg = dynamic_cast<const RouteTaskArg *>(&arg);
    if (taskArg == nullptr || taskArg->pathBytes.size() != channels_.size() ||
        taskArg->sourceOffsets.size() != channels_.size() ||
        taskArg->remoteOffsets.size() != channels_.size() ||
        (localCopy_ && taskArg->selfBytes == 0) ||
        (groupCopy_ ? 9 : localCopy_ ? 6 : 4) + 3 * channels_.size() > 48) {
        return {};
    }
    std::vector<uint64_t> args = {taskArg->inputAddr, taskArg->outputAddr,
                                  taskArg->inputToken, taskArg->outputToken};
    if (localCopy_) {
        args.push_back(taskArg->selfOffset);
        if(groupCopy_) {
            std::array<uint64_t,4> go;
            if(!BuildGroupCopyPlan(taskArg->selfBytes,&go)) return {};
            args.insert(args.end(),go.begin(),go.end());
        } else args.push_back(taskArg->selfBytes);
    }
    for (size_t i = 0; i < taskArg->pathBytes.size(); ++i) {
        args.push_back(taskArg->sourceOffsets[i]);
        args.push_back(taskArg->remoteOffsets[i]);
        args.push_back(taskArg->pathBytes[i]);
    }
    return args;
}

std::unique_ptr<a5ccu::CcuKernel> CreateRouteKernel(const a5ccu::CcuKernelArg &arg)
{
    return std::unique_ptr<a5ccu::CcuKernel>(new RouteKernel(arg));
}

} // namespace a5_ccu_urma_probe
