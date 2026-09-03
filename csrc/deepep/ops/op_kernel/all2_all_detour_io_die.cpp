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
constexpr uint64_t TRANSFER_ALIGN_BYTES = 32UL;

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

__aicore__ inline uint64_t LaneBytes(uint64_t totalBytes, uint32_t laneCount, uint32_t laneIndex)
{
    const uint64_t totalUnits = totalBytes / TRANSFER_ALIGN_BYTES;
    const uint64_t baseUnits = totalUnits / laneCount;
    const uint64_t remainderUnits = totalUnits % laneCount;
    return (baseUnits + (laneIndex < remainderUnits ? 1UL : 0UL)) * TRANSFER_ALIGN_BYTES;
}

__aicore__ inline uint64_t LaneOffset(uint64_t totalBytes, uint32_t laneCount, uint32_t laneIndex)
{
    const uint64_t totalUnits = totalBytes / TRANSFER_ALIGN_BYTES;
    const uint64_t baseUnits = totalUnits / laneCount;
    const uint64_t remainderUnits = totalUnits % laneCount;
    return (static_cast<uint64_t>(laneIndex) * baseUnits +
            (static_cast<uint64_t>(laneIndex) < remainderUnits ? laneIndex : remainderUnits)) *
        TRANSFER_ALIGN_BYTES;
}

__aicore__ inline uint64_t SliceBytes(uint64_t totalBytes, uint32_t pathCount, uint32_t pathIndex)
{
    const uint32_t laneCount = pathCount + 1U;
    if (pathIndex == 0U) {
        return LaneBytes(totalBytes, laneCount, 0U) + LaneBytes(totalBytes, laneCount, 1U);
    }
    return LaneBytes(totalBytes, laneCount, pathIndex + 1U);
}

__aicore__ inline uint64_t SliceOffset(uint64_t totalBytes, uint32_t pathCount, uint32_t pathIndex)
{
    if (pathIndex == 0U) return 0UL;
    return LaneOffset(totalBytes, pathCount + 1U, pathIndex + 1U);
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

    const uint32_t blockIdx = GetBlockIdx();
    auto *context = reinterpret_cast<__gm__ HcclCombinOpParam *>(GetHcclContext<0>());
    if (context == nullptr) {
        if (blockIdx == 0U) PRINTF("[A5 window debug] HCCL context is null\n");
        return;
    }

    const uint32_t rankSize = context->rankDim;
    const uint32_t rankId = context->rankId;
    const uint32_t commRankCount = tilingData.info.commRankCount;
    const uint64_t perRankBytes = tilingData.info.perRankBytes;
    const uint64_t cellBytes = tilingData.info.cellBytes;
    const uint64_t dataRegionOffset = tilingData.info.dataRegionOffset;
    const uint64_t magic = tilingData.info.magic;
    const uint32_t pathCount = tilingData.info.pathCount;
    const uint32_t laneCount = tilingData.info.laneCount;
    const uint32_t usedAivNum = tilingData.info.usedAivNum;
    const bool debugEnable = tilingData.info.debugEnable != 0U;

    if (rankSize != tilingData.info.rankSize || rankId >= rankSize || rankSize > MAX_RANKS ||
        commRankCount != 2U || pathCount != rankSize - commRankCount + 1U ||
        laneCount != pathCount + 1U || usedAivNum != pathCount + 1U || blockIdx >= usedAivNum) return;

    GlobalTensor<int32_t> commRanks;
    commRanks.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(commRankIds));
    const int32_t selfCommIndex = CommIndex(commRanks, commRankCount, rankId);
    if (selfCommIndex < 0) return;

    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        const uint64_t windowBase = context->windowsIn[rank];
        if (windowBase == 0UL) {
            if (blockIdx == 0U) {
                PRINTF("[A5 window debug] caller_rank=%u window_rank=%u windowsIn=0x%lx INVALID\n",
                       rankId, rank, windowBase);
            }
            return;
        }
        if (debugEnable && blockIdx == 0U) {
            const uint64_t dataBase = windowBase + dataRegionOffset;
            PRINTF("[A5 window debug] caller_rank=%u window_rank=%u windowsIn=0x%lx data_base=0x%lx\n",
                   rankId, rank, windowBase, dataBase);
        }
    }

    auto *send = reinterpret_cast<__gm__ uint8_t *>(sendData);
    auto *recv = reinterpret_cast<__gm__ uint8_t *>(recvData);

    // A dedicated block handles the local peer block while every remaining
    // block owns one communication path (direct or one explicit relay).
    if (blockIdx == 0U) {
        CopyBytes(recv + static_cast<uint64_t>(selfCommIndex) * perRankBytes,
                  send + static_cast<uint64_t>(selfCommIndex) * perRankBytes,
                  perRankBytes);
        return;
    }

    const uint32_t path = blockIdx - 1U;
    const uint32_t peerIndex = selfCommIndex == 0 ? 1U : 0U;
    const uint32_t peerRank = static_cast<uint32_t>(commRanks.GetValue(peerIndex));
    const uint64_t bytes = SliceBytes(perRankBytes, pathCount, path);
    const uint64_t sliceOffset = SliceOffset(perRankBytes, pathCount, path);

    // Outbound: this block writes its weighted slice and publishes its flag.
    const uint32_t writeMemRank = PathMemoryRank(commRanks, commRankCount, rankSize, peerRank, path);
    if (writeMemRank >= rankSize) return;
    const uint64_t writeCellIndex = static_cast<uint64_t>(rankId) * rankSize + peerRank;
    auto *writeCell = reinterpret_cast<__gm__ uint8_t *>(context->windowsIn[writeMemRank] +
        dataRegionOffset + writeCellIndex * cellBytes);
    if (debugEnable) {
        PRINTF("[A5 path debug] rank=%u block=%u path=%u mem_rank=%u offset=%lu bytes=%lu\n",
               rankId, blockIdx, path, writeMemRank, sliceOffset, bytes);
    }
    CopyBytes(writeCell + CELL_FLAG_BYTES,
              send + static_cast<uint64_t>(peerIndex) * perRankBytes + sliceOffset, bytes);
    WriteFlag(reinterpret_cast<__gm__ uint64_t *>(writeCell + rankId * FLAG_STRIDE_BYTES), magic);

    // Inbound: the same path block waits for the peer and reads its slice.
    const uint32_t readMemRank = PathMemoryRank(commRanks, commRankCount, rankSize, rankId, path);
    if (readMemRank >= rankSize) return;
    const uint64_t readCellIndex = static_cast<uint64_t>(peerRank) * rankSize + rankId;
    auto *readCell = reinterpret_cast<__gm__ uint8_t *>(context->windowsIn[readMemRank] +
        dataRegionOffset + readCellIndex * cellBytes);
    WaitFlag(reinterpret_cast<__gm__ uint64_t *>(readCell + peerRank * FLAG_STRIDE_BYTES), magic);
    CopyBytes(recv + static_cast<uint64_t>(peerIndex) * perRankBytes + sliceOffset,
              readCell + CELL_FLAG_BYTES, bytes);
}
