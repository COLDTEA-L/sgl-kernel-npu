#include <cstddef>

#include "kernel_operator.h"
#include "lib/hccl/hccl.h"
#include "common.h"
#include "all2_all_detour_io_die_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t MAX_CCU_RANKS = 32U;
constexpr uint32_t PARAM_ARRAY_COUNT = 4U;
}

extern "C" __global__ __aicore__ void all2_all_detour_io_die(
    GM_ADDR sendData, GM_ADDR commRankIds, GM_ADDR recvData, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(All2AllDetourIoDieTilingData);
    GET_TILING_DATA_WITH_STRUCT(All2AllDetourIoDieTilingData, tilingData, tiling);

    const uint32_t rankSize = tilingData.info.rankSize;
    const uint32_t rankId = tilingData.info.rankId;
    const uint32_t commRankCount = tilingData.info.commRankCount;
    const uint64_t perRankBytes = tilingData.info.perRankBytes;

    __gm__ HcclCombineOpParam *context =
        reinterpret_cast<__gm__ HcclCombineOpParam *>(GetHcclContext<0>());
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(context), &tilingData);
    hccl.SetCcTilingV2(offsetof(All2AllDetourIoDieTilingData, mc2CcTiling));
    SyncAll<true>();

    if (GetBlockIdx() == 0) {
        GlobalTensor<int32_t> commRanks;
        commRanks.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(commRankIds));

        bool selfParticipates = false;
        for (uint32_t i = 0; i < commRankCount; ++i) {
            if (static_cast<uint32_t>(commRanks.GetValue(i)) == rankId) {
                selfParticipates = true;
                break;
            }
        }

        TPipe pipe;
        TBuf<TPosition::VECCALC> paramBuf;
        pipe.InitBuffer(paramBuf, PARAM_ARRAY_COUNT * MAX_CCU_RANKS * sizeof(uint64_t));
        LocalTensor<uint64_t> params = paramBuf.Get<uint64_t>();

        const uint32_t sendCountsBase = 0U;
        const uint32_t sendDisplsBase = MAX_CCU_RANKS;
        const uint32_t recvCountsBase = 2U * MAX_CCU_RANKS;
        const uint32_t recvDisplsBase = 3U * MAX_CCU_RANKS;

        for (uint32_t peer = 0; peer < rankSize; ++peer) {
            bool peerParticipates = false;
            uint32_t peerCommIndex = 0U;
            for (uint32_t i = 0; i < commRankCount; ++i) {
                if (static_cast<uint32_t>(commRanks.GetValue(i)) == peer) {
                    peerParticipates = true;
                    peerCommIndex = i;
                    break;
                }
            }

            const uint64_t count = (selfParticipates && peerParticipates) ? perRankBytes : 0UL;
            const uint64_t displacement = static_cast<uint64_t>(peerCommIndex) * perRankBytes;
            params.SetValue(sendCountsBase + peer, count);
            params.SetValue(sendDisplsBase + peer, displacement);
            params.SetValue(recvCountsBase + peer, count);
            params.SetValue(recvDisplsBase + peer, displacement);
        }

        const uint64_t paramAddr = params.GetPhyAddr();
        HcclHandle handle = hccl.AlltoAllV<true>(
            sendData,
            reinterpret_cast<void *>(paramAddr + sendCountsBase * sizeof(uint64_t)),
            reinterpret_cast<void *>(paramAddr + sendDisplsBase * sizeof(uint64_t)),
            HcclDataType::HCCL_DATA_TYPE_INT8,
            recvData,
            reinterpret_cast<void *>(paramAddr + recvCountsBase * sizeof(uint64_t)),
            reinterpret_cast<void *>(paramAddr + recvDisplsBase * sizeof(uint64_t)),
            HcclDataType::HCCL_DATA_TYPE_INT8);
        hccl.Wait(handle);
    }

    SyncAll<true>();
    if (GetBlockIdx() == 0) {
        hccl.Finalize();
    }
}
