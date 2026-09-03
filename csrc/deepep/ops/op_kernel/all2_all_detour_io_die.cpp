#include "kernel_operator.h"
#include "data_copy.h"
#include "moe_distribute_base.h"
#include "all2_all_detour_io_die_tiling.h"

using namespace AscendC;

namespace {
constexpr uint64_t CELL_FLAG_BYTES = 4096UL;
constexpr uint64_t FLAG_STRIDE_BYTES = 32UL;
constexpr uint32_t COPY_CHUNK_BYTES = 64U * 1024U;
constexpr uint32_t MAX_RANKS = 32U;

__aicore__ inline bool IsCommRank(
    const GlobalTensor<int32_t> &commRanks, uint32_t commRankCount, uint32_t candidate)
{
    for (uint32_t i = 0; i < commRankCount; ++i) {
        if (static_cast<uint32_t>(commRanks.GetValue(i)) == candidate) return true;
    }
    return false;
}

__aicore__ inline int32_t CommIndex(
    const GlobalTensor<int32_t> &commRanks, uint32_t commRankCount, uint32_t candidate)
{
    for (uint32_t i = 0; i < commRankCount; ++i) {
        if (static_cast<uint32_t>(commRanks.GetValue(i)) == candidate) return static_cast<int32_t>(i);
    }
    return -1;
}

__aicore__ inline uint32_t PathMemoryRank(
    const GlobalTensor<int32_t> &commRanks, uint32_t commRankCount,
    uint32_t rankSize, uint32_t directDst, uint32_t pathIndex)
{
    if (pathIndex == 0U) return directDst;
    uint32_t detourIndex = 1U;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        if (IsCommRank(commRanks, commRankCount, rank)) continue;
        if (detourIndex == pathIndex) return rank;
        ++detourIndex;
    }
    return rankSize;
}

__aicore__ inline uint64_t SliceBytes(uint64_t totalBytes, uint32_t pathCount, uint32_t pathIndex)
{
    const uint64_t base = totalBytes / pathCount;
    const uint64_t remainder = totalBytes % pathCount;
    return base + (pathIndex < remainder ? 1UL : 0UL);
}

__aicore__ inline uint64_t SliceOffset(uint64_t totalBytes, uint32_t pathCount, uint32_t pathIndex)
{
    const uint64_t base = totalBytes / pathCount;
    const uint64_t remainder = totalBytes % pathCount;
    return static_cast<uint64_t>(pathIndex) * base +
        (static_cast<uint64_t>(pathIndex) < remainder ? pathIndex : remainder);
}

__aicore__ inline void CopyBytes(__gm__ uint8_t *dst, __gm__ uint8_t *src, uint64_t bytes)
{
    if (bytes == 0UL) return;
    __ubuf__ uint8_t *copyUb = reinterpret_cast<__ubuf__ uint8_t *>(get_imm(0));
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (uint64_t offset = 0; offset < bytes;) {
        const uint32_t current = static_cast<uint32_t>(
            bytes - offset > COPY_CHUNK_BYTES ? COPY_CHUNK_BYTES : bytes - offset);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        CpGM2UB(copyUb, src + offset, current);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        CpUB2GM(dst + offset, copyUb, current);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        offset += current;
    }
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void WriteFlag(__gm__ uint64_t *address, uint64_t value)
{
    __ubuf__ uint64_t *flagUb = reinterpret_cast<__ubuf__ uint64_t *>(get_imm(COPY_CHUNK_BYTES));
    *flagUb = value;
    AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
    CpUB2GM(address, flagUb, sizeof(uint64_t));
    AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void WaitFlag(__gm__ uint64_t *address, uint64_t expected)
{
    __ubuf__ uint64_t *flagUb = reinterpret_cast<__ubuf__ uint64_t *>(get_imm(COPY_CHUNK_BYTES));
    uint64_t value = 0UL;
    do {
        CpGM2UB(flagUb, address, sizeof(uint64_t));
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        value = *flagUb;
    } while (value != expected);
    pipe_barrier(PIPE_ALL);
}
}  // namespace

// A5 AIV+URMA relay validation:
//   communication source --MTE write--> relay HBM window
//   communication destination --MTE read--> relay HBM window
// Every non-communication rank is used as one explicit relay path.  Relay
// ranks only expose their MTE window; they execute no copy loop themselves.
extern "C" __global__ __aicore__ void all2_all_detour_io_die(
    GM_ADDR sendData, GM_ADDR commRankIds, GM_ADDR recvData, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(All2AllDetourIoDieTilingData);
    GET_TILING_DATA_WITH_STRUCT(All2AllDetourIoDieTilingData, tilingData, tiling);

    if (GetBlockIdx() != 0) return;
    auto *context = reinterpret_cast<__gm__ HcclCombinOpParam *>(GetHcclContext<0>());
    if (context == nullptr) {
        PRINTF("[A5 window debug] HCCL context is null\n");
        return;
    }

    const uint32_t rankSize = context->rankDim;
    const uint32_t rankId = context->rankId;
    const uint32_t commRankCount = tilingData.info.commRankCount;
    const uint64_t perRankBytes = tilingData.info.perRankBytes;
    const uint64_t cellBytes = tilingData.info.cellBytes;
    const uint64_t dataRegionOffset = tilingData.info.dataRegionOffset;
    const uint64_t magic = tilingData.info.magic;

    if (rankSize != tilingData.info.rankSize || rankId >= rankSize || rankSize > MAX_RANKS) return;

    GlobalTensor<int32_t> commRanks;
    commRanks.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(commRankIds));
    const int32_t selfCommIndex = CommIndex(commRanks, commRankCount, rankId);
    if (selfCommIndex < 0) return;

    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        const uint64_t windowBase = context->windowsIn[rank];
        if (windowBase == 0UL) {
            PRINTF("[A5 window debug] caller_rank=%u window_rank=%u windowsIn=0x%lx INVALID\n",
                   rankId, rank, windowBase);
            return;
        }
        const uint64_t dataBase = windowBase + dataRegionOffset;
        PRINTF("[A5 window debug] caller_rank=%u window_rank=%u windowsIn=0x%lx data_base=0x%lx\n",
               rankId, rank, windowBase, dataBase);
    }

    auto *send = reinterpret_cast<__gm__ uint8_t *>(sendData);
    auto *recv = reinterpret_cast<__gm__ uint8_t *>(recvData);

    // The block addressed to self never enters a communication window.
    CopyBytes(recv + static_cast<uint64_t>(selfCommIndex) * perRankBytes,
              send + static_cast<uint64_t>(selfCommIndex) * perRankBytes,
              perRankBytes);

    const uint32_t pathCount = rankSize - commRankCount + 1U;

    // Phase 1: write the direct slice and every explicitly selected relay.
    for (uint32_t dstIndex = 0; dstIndex < commRankCount; ++dstIndex) {
        const uint32_t dstRank = static_cast<uint32_t>(commRanks.GetValue(dstIndex));
        if (dstRank == rankId) continue;
        for (uint32_t path = 0; path < pathCount; ++path) {
            const uint32_t memRank = PathMemoryRank(commRanks, commRankCount, rankSize, dstRank, path);
            if (memRank >= rankSize) return;
            const uint64_t bytes = SliceBytes(perRankBytes, pathCount, path);
            const uint64_t inputOffset = SliceOffset(perRankBytes, pathCount, path);
            const uint64_t cellIndex = static_cast<uint64_t>(rankId) * rankSize + dstRank;
            auto *cell = reinterpret_cast<__gm__ uint8_t *>(context->windowsIn[memRank] +
                dataRegionOffset + cellIndex * cellBytes);
            CopyBytes(cell + CELL_FLAG_BYTES, send + static_cast<uint64_t>(dstIndex) * perRankBytes + inputOffset,
                      bytes);
            auto *flag = reinterpret_cast<__gm__ uint64_t *>(cell + rankId * FLAG_STRIDE_BYTES);
            WriteFlag(flag, magic);
        }
    }

    // Phase 2: wait for the peer's writes and read the same relay windows.
    for (uint32_t srcIndex = 0; srcIndex < commRankCount; ++srcIndex) {
        const uint32_t srcRank = static_cast<uint32_t>(commRanks.GetValue(srcIndex));
        if (srcRank == rankId) continue;
        for (uint32_t path = 0; path < pathCount; ++path) {
            const uint32_t memRank = PathMemoryRank(commRanks, commRankCount, rankSize, rankId, path);
            if (memRank >= rankSize) return;
            const uint64_t bytes = SliceBytes(perRankBytes, pathCount, path);
            const uint64_t outputOffset = SliceOffset(perRankBytes, pathCount, path);
            const uint64_t cellIndex = static_cast<uint64_t>(srcRank) * rankSize + rankId;
            auto *cell = reinterpret_cast<__gm__ uint8_t *>(context->windowsIn[memRank] +
                dataRegionOffset + cellIndex * cellBytes);
            auto *flag = reinterpret_cast<__gm__ uint64_t *>(cell + srcRank * FLAG_STRIDE_BYTES);
            WaitFlag(flag, magic);
            CopyBytes(recv + static_cast<uint64_t>(srcIndex) * perRankBytes + outputOffset,
                      cell + CELL_FLAG_BYTES, bytes);
        }
    }
}
