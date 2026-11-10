#include "alltoallv_kernel.h"
#include "../common/path_layout.h"
namespace a5_ccu_urma_probe {
namespace CcuRep = a5ccu::CcuRep;
namespace {
constexpr uint32_t ADDRESS_SLOT = 0, TOKEN_SLOT = 1, DONE_SLOT = 2, MASK = 1;
#define V_CHECK(call) do { const auto s = (call); if (s != HCCL_SUCCESS) return s; } while (0)
}
a5ccu::CcuKernelSignature AllToAllVKernelArg::GetKernelSignature() const
{
    a5ccu::CcuKernelSignature signature;
    signature.Append(MakePathKernelSignature("A5CcuDirectAllToAllV8BaselineV1", {}, channels));
    return signature;
}
HcclResult AllToAllVKernel::Algorithm()
{
    if (channels_.size() != 7 || 10 + 3 * channels_.size() > 48) return HCCL_E_PARA;
    auto input = CreateVariable(), output = CreateVariable();
    auto inputToken = CreateVariable(), outputToken = CreateVariable();
    auto selfSource = CreateVariable(), selfDestination = CreateVariable();
    Load(input); Load(output); Load(inputToken); Load(outputToken);
    Load(selfSource); Load(selfDestination);
    std::array<a5ccu::CcuRep::Variable, 4> go;
    // Public Variable default construction already allocates an XN resource.
    for (auto &v : go) Load(v);
    std::vector<a5ccu::CcuRep::Variable> source(7), destination(7), bytes(7), remoteOutput(7), remoteToken(7);
    std::vector<a5ccu::CcuRep::CompletedEvent> events(7);
    for (size_t i = 0; i < 7; ++i) {
        Load(source[i]); Load(destination[i]); Load(bytes[i]);
        V_CHECK(CreateVariable(channels_[i], ADDRESS_SLOT, &remoteOutput[i]));
        V_CHECK(CreateVariable(channels_[i], TOKEN_SLOT, &remoteToken[i]));
        V_CHECK(NotifyRecord(channels_[i], ADDRESS_SLOT, ADDRESS_SLOT, output, MASK));
        V_CHECK(NotifyRecord(channels_[i], TOKEN_SLOT, TOKEN_SLOT, outputToken, MASK));
        events[i] = CreateCompletedEvent();
    }
    // Publish EVERY peer before any wait; zero-length directions still take
    // part in both metadata exchange and the final per-channel handshake.
    for (auto channel : channels_) {
        V_CHECK(NotifyWait(channel, ADDRESS_SLOT, MASK));
        V_CHECK(NotifyWait(channel, TOKEN_SLOT, MASK));
    }
    for (size_t i = 0; i < 7; ++i) {
        auto src = CreateLocalAddr(); src.addr = input; src.addr += source[i]; src.token = inputToken;
        auto dst = CreateRemoteAddr(); dst.addr = remoteOutput[i];
        dst.addr += destination[i]; dst.token = remoteToken[i];
        CCU_IF(bytes[i] != 0) {
            V_CHECK(WriteNb(channels_[i], dst, src, bytes[i], events[i]));
        }
    }
    auto src = CreateLocalAddr(), dst = CreateLocalAddr();
    src.addr = input; src.addr += selfSource; src.token = inputToken;
    dst.addr = output; dst.addr += selfDestination; dst.token = outputToken;
    // An all-zero go tuple performs no copy, including an empty self row.
    V_CHECK(GroupedLocalCopy(dst, src, go));
    for (size_t i = 0; i < 7; ++i) {
        CCU_IF(bytes[i] != 0) { V_CHECK(WaitEvent(events[i])); }
    }
    for (auto channel : channels_) V_CHECK(NotifyRecord(channel, DONE_SLOT, MASK));
    for (auto channel : channels_) V_CHECK(NotifyWait(channel, DONE_SLOT, MASK));
    return HCCL_SUCCESS;
}
std::vector<uint64_t> AllToAllVKernel::GeneArgs(const a5ccu::CcuTaskArg &arg)
{
    const auto *task = dynamic_cast<const AllToAllVTaskArg *>(&arg);
    if (!task || task->layout.sourceOffsets.size() != 7 || task->layout.remoteOffsets.size() != 7 ||
        task->layout.pathBytes.size() != 7) return {};
    const auto &l = task->layout;
    std::array<uint64_t, 4> go{};
    if (l.selfBytes && !BuildGroupCopyPlan(l.selfBytes, &go)) return {};
    std::vector<uint64_t> args{task->input, task->output, task->inputToken, task->outputToken,
        l.selfSourceOffset, l.selfDestinationOffset};
    args.insert(args.end(), go.begin(), go.end());
    for (size_t i = 0; i < 7; ++i) {
        args.push_back(l.sourceOffsets[i]); args.push_back(l.remoteOffsets[i]); args.push_back(l.pathBytes[i]);
    }
    return args;
}
std::unique_ptr<a5ccu::CcuKernel> CreateAllToAllVKernel(const a5ccu::CcuKernelArg &arg)
{ return std::unique_ptr<a5ccu::CcuKernel>(new AllToAllVKernel(arg)); }
#undef V_CHECK
} // namespace a5_ccu_urma_probe
