#ifndef A5_CCU_URMA_COMMAND_BLOCK_WORKER_KERNEL_H
#define A5_CCU_URMA_COMMAND_BLOCK_WORKER_KERNEL_H

#include <hcomm/ccu/ccu_kernel.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace a5_ccu_urma_probe {

// Keep this layout in sync with
// csrc/deepep/ops/op_kernel/ccu_hbm_command_puncture_tiling.h.
// Every entry is one 64-bit word so the CCU can use CcuLoadVar/CcuStoreVar.
enum CommandBlockWord : uint64_t {
    COMMAND_MAGIC = 0,
    COMMAND_OPCODE = 1,
    COMMAND_COMPLETION = 2,
    COMMAND_STATUS = 3,
    COMMAND_SEND_ADDR = 4,
    COMMAND_RECV_ADDR = 5,
    COMMAND_INPUT_TOKEN = 6,
    COMMAND_OUTPUT_TOKEN = 7,
    COMMAND_PER_PEER_BYTES = 8,
    COMMAND_SELF_SOURCE_OFFSET = 9,
    COMMAND_SELF_DESTINATION_OFFSET = 10,
    COMMAND_SELF_BYTES = 11,
    COMMAND_PATH_COUNT = 12,
    COMMAND_PATH_MASK = 13,
    COMMAND_SOURCE_OFFSETS = 14,
    COMMAND_MAX_PATHS = 8,
    COMMAND_REMOTE_OFFSETS = COMMAND_SOURCE_OFFSETS + COMMAND_MAX_PATHS,
    COMMAND_PATH_BYTES = COMMAND_REMOTE_OFFSETS + COMMAND_MAX_PATHS,
    COMMAND_WORDS = COMMAND_PATH_BYTES + COMMAND_MAX_PATHS,
};

constexpr uint64_t COMMAND_BLOCK_MAGIC = 0x41354343554d424cULL; // "A5CCUMBL"
constexpr uint64_t COMMAND_OPCODE_EXECUTE = 1;
constexpr uint64_t COMMAND_OPCODE_STOP = 2;
constexpr uint64_t COMMAND_STATUS_SUCCESS = 0;

// Registration-only probe modes.  These deliberately split the persistent
// worker into orthogonal instruction groups so an A5 machine can identify
// which group is rejected by HcclCcuKernelRegister without launching it.
enum class CommandWorkerRegisterMode : uint32_t {
    FULL = 0,
    HBM_ONCE = 1,
    LOOP_ONLY = 2,
    LOOP_HBM = 3,
};

class CommandBlockWorkerKernelArg : public hcomm::CcuKernelArg {
public:
    CommandBlockWorkerKernelArg(const std::vector<ChannelHandle> &channels,
                                const std::vector<uint32_t> &routeIndices,
                                const std::vector<uint32_t> &weights,
                                uint64_t commandBlockAddress,
                                CommandWorkerRegisterMode registerMode);
    hcomm::CcuKernelSignature GetKernelSignature() const override;
    uint64_t GetCommandBlockAddress() const { return commandBlockAddress_; }
    CommandWorkerRegisterMode GetRegisterMode() const { return registerMode_; }

private:
    std::vector<uint32_t> routeIndices_;
    std::vector<uint32_t> weights_;
    uint64_t commandBlockAddress_ = 0;
    CommandWorkerRegisterMode registerMode_ = CommandWorkerRegisterMode::FULL;
};

class CommandBlockWorkerTaskArg : public hcomm::CcuTaskArg {
public:
    explicit CommandBlockWorkerTaskArg(uint64_t commandBlockAddr)
        : commandBlockAddr(commandBlockAddr) {}
    uint64_t commandBlockAddr;
};

class CommandBlockWorkerKernel : public hcomm::CcuKernel {
public:
    explicit CommandBlockWorkerKernel(const hcomm::CcuKernelArg &arg);

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    uint64_t commandBlockAddress_ = 0;
    CommandWorkerRegisterMode registerMode_ = CommandWorkerRegisterMode::FULL;
};

std::unique_ptr<hcomm::CcuKernel> CreateCommandBlockWorkerKernel(
    const hcomm::CcuKernelArg &arg);

} // namespace a5_ccu_urma_probe

#endif
