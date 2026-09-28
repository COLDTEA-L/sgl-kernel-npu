#include "command_block_worker_kernel.h"

#include <cstdio>

namespace a5_ccu_urma_probe {
namespace {
constexpr uint32_t OUTPUT_VAR_INDEX = 0;
constexpr uint32_t TOKEN_VAR_INDEX = 1;
constexpr uint32_t OUTPUT_NOTIFY_INDEX = 0;
constexpr uint32_t TOKEN_NOTIFY_INDEX = 1;
constexpr uint32_t COMPLETION_NOTIFY_INDEX = 2;
constexpr uint32_t NOTIFY_MASK = 1;

#define CCU_KERNEL_CHECK(expression) do { \
    HcclResult status = (expression); \
    if (status != HCCL_SUCCESS) { \
        std::fprintf(stderr, \
            "COMMAND_BLOCK_REGISTER_TRACE phase=primitive_failed line=%d status=%d\n", \
            __LINE__, static_cast<int>(status)); \
        std::fflush(stderr); \
        return status; \
    } \
} while (0)

uint64_t AddressAt(uint64_t base, uint64_t word)
{
    return base + word * sizeof(uint64_t);
}

const char *RegisterModeName(CommandWorkerRegisterMode mode)
{
    switch (mode) {
        case CommandWorkerRegisterMode::FULL: return "full";
        case CommandWorkerRegisterMode::HBM_ONCE: return "hbm_once";
        case CommandWorkerRegisterMode::LOOP_ONLY: return "loop_only";
        case CommandWorkerRegisterMode::LOOP_HBM: return "loop_hbm";
    }
    return "invalid";
}
} // namespace

CommandBlockWorkerKernelArg::CommandBlockWorkerKernelArg(
    const std::vector<ChannelHandle> &inputChannels,
    const std::vector<uint32_t> &routeIndices,
    const std::vector<uint32_t> &weights,
    uint64_t commandBlockAddress,
    CommandWorkerRegisterMode registerMode)
    : routeIndices_(routeIndices), weights_(weights),
      commandBlockAddress_(commandBlockAddress), registerMode_(registerMode)
{
    channels = inputChannels;
}

hcomm::CcuKernelSignature CommandBlockWorkerKernelArg::GetKernelSignature() const
{
    hcomm::CcuKernelSignature signature;
    signature.Append("A5CcuHbmCommandBlockWorkerV1");
    for (size_t i = 0; i < routeIndices_.size(); ++i) {
        signature.Append(routeIndices_[i]);
        signature.Append(weights_[i]);
    }
    signature.Append(commandBlockAddress_);
    signature.Append(static_cast<uint32_t>(registerMode_));
    return signature;
}

CommandBlockWorkerKernel::CommandBlockWorkerKernel(const hcomm::CcuKernelArg &arg)
    : hcomm::CcuKernel(arg)
{
    const auto *workerArg = dynamic_cast<const CommandBlockWorkerKernelArg *>(&arg);
    if (workerArg != nullptr) {
        commandBlockAddress_ = workerArg->GetCommandBlockAddress();
        registerMode_ = workerArg->GetRegisterMode();
    }
}

HcclResult CommandBlockWorkerKernel::Algorithm()
{
    if (channels_.empty() || channels_.size() > COMMAND_MAX_PATHS) {
        return HCCL_E_PARA;
    }

    if (commandBlockAddress_ == 0U) return HCCL_E_PARA;
    std::printf("COMMAND_BLOCK_REGISTER_TRACE phase=algorithm_begin mode=%s paths=%zu "
                "command_block=0x%lx\n", RegisterModeName(registerMode_),
                channels_.size(), static_cast<unsigned long>(commandBlockAddress_));
    std::fflush(stdout);
    using namespace hcomm;
    using hcomm::CcuRep::CompletedEvent;
    using hcomm::CcuRep::LocalAddr;
    using hcomm::CcuRep::RemoteAddr;
    using hcomm::CcuRep::Variable;

    // GeneArgs() supplies exactly one task argument.  Every working HCOMM CCU
    // kernel in this repository consumes each task argument with Load() while
    // building the instruction template.  Omitting this Load made all probe
    // modes share an invalid task-argument layout and obscured the actual
    // HBM/control-flow capability result.
    Variable runtimeCommandBlock;
    Load(runtimeCommandBlock);
    std::printf("COMMAND_BLOCK_REGISTER_TRACE phase=task_arg_ready mode=%s\n",
                RegisterModeName(registerMode_));
    std::fflush(stdout);

    // Registration-only capability probes.  The host never launches these
    // kernels when A5_CCU_WORKER_REGISTER_ONLY=1; their sole purpose is to
    // make the registration boundary attributable to one instruction group.
    if (registerMode_ == CommandWorkerRegisterMode::HBM_ONCE) {
        Variable value;
        value = 0;
        LoadVariable(AddressAt(commandBlockAddress_, COMMAND_OPCODE), value);
        StoreVariable(value, AddressAt(commandBlockAddress_, COMMAND_STATUS));
        std::printf("COMMAND_BLOCK_REGISTER_TRACE phase=algorithm_ready mode=hbm_once\n");
        std::fflush(stdout);
        return HCCL_SUCCESS;
    }
    if (registerMode_ == CommandWorkerRegisterMode::LOOP_ONLY) {
        Variable once;
        once = 0;
        CCU_WHILE(once == 0) {
            once = 1;
        }
        std::printf("COMMAND_BLOCK_REGISTER_TRACE phase=algorithm_ready mode=loop_only\n");
        std::fflush(stdout);
        return HCCL_SUCCESS;
    }
    if (registerMode_ == CommandWorkerRegisterMode::LOOP_HBM) {
        Variable command;
        command = 0;
        CCU_WHILE(command == 0) {
            LoadVariable(AddressAt(commandBlockAddress_, COMMAND_OPCODE), command);
        }
        std::printf("COMMAND_BLOCK_REGISTER_TRACE phase=algorithm_ready mode=loop_hbm\n");
        std::fflush(stdout);
        return HCCL_SUCCESS;
    }

    std::vector<Variable> remoteOutputs(channels_.size());
    std::vector<Variable> remoteTokens(channels_.size());
    for (size_t i = 0; i < channels_.size(); ++i) {
        CCU_KERNEL_CHECK(CreateVariable(channels_[i], OUTPUT_VAR_INDEX, &remoteOutputs[i]));
        CCU_KERNEL_CHECK(CreateVariable(channels_[i], TOKEN_VAR_INDEX, &remoteTokens[i]));
    }
    std::printf("COMMAND_BLOCK_REGISTER_TRACE phase=remote_variables_ready mode=full paths=%zu\n",
                channels_.size());
    std::fflush(stdout);

    Variable stop;
    stop = 0;
    CCU_WHILE(stop == 0) {
        Variable command;
        command = 0;
        // Keep a single CCU control-flow loop.  The A5 instruction builder
        // rejects the nested WHILE form used by an earlier puncture revision
        // during HcclCcuKernelRegister.  IDLE simply performs one load and
        // advances to the next outer iteration, which is the same busy-poll
        // behaviour without nested loop metadata.
        LoadVariable(AddressAt(commandBlockAddress_, COMMAND_OPCODE), command);

        CCU_IF(command == COMMAND_OPCODE_STOP) {
            Variable status;
            status = COMMAND_STATUS_SUCCESS;
            StoreVariable(status, AddressAt(commandBlockAddress_, COMMAND_STATUS));
            command = 0;
            StoreVariable(command, AddressAt(commandBlockAddress_, COMMAND_OPCODE));
            // Publish completion last.  The AIV producer is allowed to submit
            // the next epoch as soon as it observes completion, so clearing
            // the consumed command afterwards could otherwise overwrite that
            // next submission.
            Variable completed;
            completed = 1;
            StoreVariable(completed, AddressAt(commandBlockAddress_, COMMAND_COMPLETION));
            stop = 1;
        }

        CCU_IF(command == COMMAND_OPCODE_EXECUTE) {
            Variable input;
            Variable output;
            Variable inputToken;
            Variable outputToken;
            Variable selfSourceOffset;
            Variable selfDestinationOffset;
            Variable selfBytes;
            Variable pathMask;
            LoadVariable(AddressAt(commandBlockAddress_, COMMAND_SEND_ADDR), input);
            LoadVariable(AddressAt(commandBlockAddress_, COMMAND_RECV_ADDR), output);
            LoadVariable(AddressAt(commandBlockAddress_, COMMAND_INPUT_TOKEN), inputToken);
            LoadVariable(AddressAt(commandBlockAddress_, COMMAND_OUTPUT_TOKEN), outputToken);
            LoadVariable(AddressAt(commandBlockAddress_, COMMAND_SELF_SOURCE_OFFSET), selfSourceOffset);
            LoadVariable(AddressAt(commandBlockAddress_, COMMAND_SELF_DESTINATION_OFFSET), selfDestinationOffset);
            LoadVariable(AddressAt(commandBlockAddress_, COMMAND_SELF_BYTES), selfBytes);
            LoadVariable(AddressAt(commandBlockAddress_, COMMAND_PATH_MASK), pathMask);

            // pathMask==0 is the visibility-only stage. It deliberately does
            // no copy, so a successful completion proves only AIV->HBM->CCU
            // command visibility and CCU->HBM->AIV acknowledgement.
            CCU_IF(pathMask != 0) {
                for (size_t i = 0; i < channels_.size(); ++i) {
                    CCU_KERNEL_CHECK(NotifyRecord(channels_[i], OUTPUT_NOTIFY_INDEX,
                                                  OUTPUT_VAR_INDEX, output, NOTIFY_MASK));
                    CCU_KERNEL_CHECK(NotifyRecord(channels_[i], TOKEN_NOTIFY_INDEX,
                                                  TOKEN_VAR_INDEX, outputToken, NOTIFY_MASK));
                }
                for (const ChannelHandle channel : channels_) {
                    CCU_KERNEL_CHECK(NotifyWait(channel, OUTPUT_NOTIFY_INDEX, NOTIFY_MASK));
                    CCU_KERNEL_CHECK(NotifyWait(channel, TOKEN_NOTIFY_INDEX, NOTIFY_MASK));
                }

                LocalAddr localSource = CreateLocalAddr();
                localSource.addr = input;
                localSource.addr += selfSourceOffset;
                localSource.token = inputToken;
                LocalAddr localDestination = CreateLocalAddr();
                localDestination.addr = output;
                localDestination.addr += selfDestinationOffset;
                localDestination.token = outputToken;
                CompletedEvent localEvent = CreateCompletedEvent();
                CCU_KERNEL_CHECK(LocalCopyNb(localDestination, localSource,
                                             selfBytes, localEvent));

                std::vector<CompletedEvent> remoteEvents;
                remoteEvents.reserve(channels_.size());
                for (size_t i = 0; i < channels_.size(); ++i) {
                    Variable sourceOffset;
                    Variable remoteOffset;
                    Variable pathBytes;
                    LoadVariable(AddressAt(commandBlockAddress_,
                        COMMAND_SOURCE_OFFSETS + i), sourceOffset);
                    LoadVariable(AddressAt(commandBlockAddress_,
                        COMMAND_REMOTE_OFFSETS + i), remoteOffset);
                    LoadVariable(AddressAt(commandBlockAddress_,
                        COMMAND_PATH_BYTES + i), pathBytes);
                    LocalAddr source = CreateLocalAddr();
                    source.addr = input;
                    source.addr += sourceOffset;
                    source.token = inputToken;
                    RemoteAddr destination = CreateRemoteAddr();
                    destination.addr = remoteOutputs[i];
                    destination.addr += remoteOffset;
                    destination.token = remoteTokens[i];
                    remoteEvents.push_back(CreateCompletedEvent());
                    CCU_KERNEL_CHECK(WriteNb(channels_[i], destination, source,
                                             pathBytes, remoteEvents.back()));
                }
                CCU_KERNEL_CHECK(WaitEvent(localEvent));
                for (auto &event : remoteEvents) {
                    CCU_KERNEL_CHECK(WaitEvent(event));
                }
                CCU_KERNEL_CHECK(NotifyRecord(channels_.front(),
                                              COMPLETION_NOTIFY_INDEX, NOTIFY_MASK));
                CCU_KERNEL_CHECK(NotifyWait(channels_.front(),
                                            COMPLETION_NOTIFY_INDEX, NOTIFY_MASK));
            }

            Variable status;
            status = COMMAND_STATUS_SUCCESS;
            StoreVariable(status, AddressAt(commandBlockAddress_, COMMAND_STATUS));
            command = 0;
            StoreVariable(command, AddressAt(commandBlockAddress_, COMMAND_OPCODE));
            // completion is the release flag for this command epoch and must
            // therefore be the final store performed by the worker.
            Variable completed;
            completed = 1;
            StoreVariable(completed, AddressAt(commandBlockAddress_, COMMAND_COMPLETION));
        }
    }
    std::printf("COMMAND_BLOCK_REGISTER_TRACE phase=algorithm_ready mode=full paths=%zu\n",
                channels_.size());
    std::fflush(stdout);
    return HCCL_SUCCESS;
}

std::vector<uint64_t> CommandBlockWorkerKernel::GeneArgs(
    const hcomm::CcuTaskArg &arg)
{
    const auto *taskArg = dynamic_cast<const CommandBlockWorkerTaskArg *>(&arg);
    if (taskArg == nullptr) return {};
    return {taskArg->commandBlockAddr};
}

std::unique_ptr<hcomm::CcuKernel> CreateCommandBlockWorkerKernel(
    const hcomm::CcuKernelArg &arg)
{
    return std::unique_ptr<hcomm::CcuKernel>(new CommandBlockWorkerKernel(arg));
}

} // namespace a5_ccu_urma_probe
