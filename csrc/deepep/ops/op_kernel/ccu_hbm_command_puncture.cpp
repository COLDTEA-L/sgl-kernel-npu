#include <cstdint>

#include "kernel_operator.h"
#include "ccu_hbm_command_puncture_tiling.h"

using namespace AscendC;

namespace {
constexpr uint64_t MAGIC = 0x41354343554d424cULL;
constexpr uint64_t EXECUTE = 1;
constexpr uint64_t COMPLETED = 1;
constexpr uint64_t PATH_ALIGNMENT = 256;
constexpr uint64_t WORD_MAGIC = 0;
constexpr uint64_t WORD_COMMAND = 1;
constexpr uint64_t WORD_COMPLETION = 2;
constexpr uint64_t WORD_STATUS = 3;
constexpr uint64_t WORD_SEND_ADDR = 4;
constexpr uint64_t WORD_RECV_ADDR = 5;
constexpr uint64_t WORD_PER_PEER_BYTES = 8;
constexpr uint64_t WORD_SELF_SOURCE_OFFSET = 9;
constexpr uint64_t WORD_SELF_DESTINATION_OFFSET = 10;
constexpr uint64_t WORD_SELF_BYTES = 11;
constexpr uint64_t WORD_PATH_COUNT = 12;
constexpr uint64_t WORD_PATH_MASK = 13;
constexpr uint64_t WORD_SOURCE_OFFSETS = 14;
constexpr uint64_t WORD_REMOTE_OFFSETS = WORD_SOURCE_OFFSETS + A5_COMMAND_PUNCTURE_MAX_PATHS;
constexpr uint64_t WORD_PATH_BYTES = WORD_REMOTE_OFFSETS + A5_COMMAND_PUNCTURE_MAX_PATHS;

__aicore__ inline void FlushLine(__gm__ uint64_t *address)
{
    GlobalTensor<uint64_t> line;
    line.SetGlobalBuffer(address);
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(line);
}
} // namespace

extern "C" __global__ __aicore__ void ccu_hbm_command_puncture(
    GM_ADDR sendData, GM_ADDR recvData, GM_ADDR commandBlock, GM_ADDR ack,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(CcuHbmCommandPunctureTilingData);
    GET_TILING_DATA_WITH_STRUCT(CcuHbmCommandPunctureTilingData, tilingData, tiling);
    // Tiling requests one AIV block and zero AIC blocks. ASCEND_IS_AIV is a
    // statement-style compiler macro on this CANN line, not a bool expression.
    if (GetBlockIdx() != 0) return;

    auto *words = reinterpret_cast<__gm__ uint64_t *>(commandBlock);
    GlobalTensor<uint64_t> command;
    command.SetGlobalBuffer(words);
    GlobalTensor<int32_t> ackTensor;
    ackTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ack));

    // The producer owns command/completion. Clear stale completion before
    // publishing the next command so ACLGraph replay can reuse the same block.
    command(WORD_COMMAND) = 0;
    command(WORD_COMPLETION) = 0;
    command(WORD_STATUS) = UINT64_MAX;
    command(WORD_MAGIC) = MAGIC;
    command(WORD_SEND_ADDR) = reinterpret_cast<uint64_t>(sendData);
    command(WORD_RECV_ADDR) = reinterpret_cast<uint64_t>(recvData);
    command(WORD_PER_PEER_BYTES) = tilingData.info.perPeerBytes;
    command(WORD_SELF_SOURCE_OFFSET) =
        static_cast<uint64_t>(tilingData.info.rankId) * tilingData.info.perPeerBytes;
    command(WORD_SELF_DESTINATION_OFFSET) =
        static_cast<uint64_t>(tilingData.info.rankId) * tilingData.info.perPeerBytes;
    command(WORD_SELF_BYTES) = tilingData.info.transfer ?
        tilingData.info.perPeerBytes : 0;
    command(WORD_PATH_COUNT) = tilingData.info.pathCount;
    command(WORD_PATH_MASK) = tilingData.info.transfer ?
        ((1ULL << tilingData.info.pathCount) - 1ULL) : 0ULL;

    uint64_t totalWeight = 0;
    for (uint32_t i = 0; i < tilingData.info.pathCount; ++i) {
        totalWeight += tilingData.info.pathWeights[i];
    }
    uint64_t assigned = 0;
    const uint64_t peer = 1U - tilingData.info.rankId;
    for (uint32_t i = 0; i < A5_COMMAND_PUNCTURE_MAX_PATHS; ++i) {
        uint64_t bytes = 0;
        if (i < tilingData.info.pathCount) {
            bytes = tilingData.info.perPeerBytes - assigned;
            if (i + 1U != tilingData.info.pathCount) {
                bytes = (tilingData.info.perPeerBytes *
                         tilingData.info.pathWeights[i] / totalWeight) /
                        PATH_ALIGNMENT * PATH_ALIGNMENT;
            }
            command(WORD_SOURCE_OFFSETS + i) =
                peer * tilingData.info.perPeerBytes + assigned;
            command(WORD_REMOTE_OFFSETS + i) =
                static_cast<uint64_t>(tilingData.info.rankId) *
                    tilingData.info.perPeerBytes + assigned;
            assigned += bytes;
        } else {
            command(WORD_SOURCE_OFFSETS + i) = 0;
            command(WORD_REMOTE_OFFSETS + i) = 0;
        }
        command(WORD_PATH_BYTES + i) = tilingData.info.transfer ? bytes : 0;
    }

    PipeBarrier<PIPE_ALL>();
    FlushLine(words + 0);
    FlushLine(words + 8);
    FlushLine(words + 16);
    FlushLine(words + 24);
    FlushLine(words + 32);
    PipeBarrier<PIPE_ALL>();
    command(WORD_COMMAND) = EXECUTE;
    PipeBarrier<PIPE_ALL>();
    FlushLine(words + WORD_COMMAND);

    uint64_t completion = 0;
    uint64_t spins = 0;
    for (; spins < tilingData.info.maxSpinCount; ++spins) {
        FlushLine(words + WORD_COMPLETION);
        PipeBarrier<PIPE_ALL>();
        completion = command(WORD_COMPLETION);
        if (completion == COMPLETED) break;
    }
    FlushLine(words + WORD_STATUS);
    PipeBarrier<PIPE_ALL>();
    const uint64_t workerStatus = command(WORD_STATUS);
    ackTensor(0) = (completion == COMPLETED && workerStatus == 0) ? 0 : -1;
    PipeBarrier<PIPE_ALL>();
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(ackTensor);
}
