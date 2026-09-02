#include <cstddef>

#include "kernel_operator.h"
#include "lib/hccl/hccl.h"
#include "common.h"
#include "all2_all_detour_io_die_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t MAX_CCU_RANKS = 32U;
}

extern "C" __global__ __aicore__ void all2_all_detour_io_die(
    GM_ADDR sendData, GM_ADDR commRankIds, GM_ADDR recvData, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(All2AllDetourIoDieTilingData);
    GET_TILING_DATA_WITH_STRUCT(All2AllDetourIoDieTilingData, tilingData, tiling);

    const uint32_t commRankCount = tilingData.info.commRankCount;
    const uint64_t perRankBytes = tilingData.info.perRankBytes;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ HcclCombineOpParam *context =
        reinterpret_cast<__gm__ HcclCombineOpParam *>(GetHcclContext<0>());
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(context), &tilingData);
    hccl.SetCcTilingV2(offsetof(All2AllDetourIoDieTilingData, mc2CcTiling));

    if ASCEND_IS_AIV {
        // Use the communicator values populated by InitV2. This is the same
        // source of truth used by CANN's AlltoAllvWrite sample and avoids a
        // rank mismatch between operator attributes and the HCCL context.
        const uint32_t rankSize = hccl.GetRankDim();
        const uint32_t rankId = hccl.GetRankId();
        GlobalTensor<int32_t> commRanks;
        commRanks.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(commRankIds));

        bool selfParticipates = false;
        for (uint32_t i = 0; i < commRankCount; ++i) {
            if (static_cast<uint32_t>(commRanks.GetValue(i)) == rankId) {
                selfParticipates = true;
                break;
            }
        }

        // AlltoAllvWrite submits HCCL_CMD_HALF_ALLTOALLV. That command is
        // intended for the dispatch/combine window protocol and is not a
        // complete AllToAllV collective. This validation operator instead
        // submits HCCL_CMD_ALLTOALLV, for which the A5 runtime registers the
        // CCU Mesh1D/2Die executors. INT8 makes counts/displacements bytes.
        SyncAll<true>();
        if (GetBlockIdx() == 0) {
            uint64_t sendCounts[MAX_CCU_RANKS] = {0UL};
            uint64_t sendDisplacements[MAX_CCU_RANKS] = {0UL};
            uint64_t recvCounts[MAX_CCU_RANKS] = {0UL};
            uint64_t recvDisplacements[MAX_CCU_RANKS] = {0UL};

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
                const bool shouldSend = selfParticipates && peerParticipates;
                const uint64_t peerOffset = static_cast<uint64_t>(peerCommIndex) * perRankBytes;
                sendCounts[peer] = shouldSend ? perRankBytes : 0UL;
                sendDisplacements[peer] = peerOffset;
                recvCounts[peer] = shouldSend ? perRankBytes : 0UL;
                recvDisplacements[peer] = peerOffset;
            }

            HcclHandle handle = hccl.AlltoAllV<true>(
                sendData,
                sendCounts,
                sendDisplacements,
                HcclDataType::HCCL_DATA_TYPE_INT8,
                recvData,
                recvCounts,
                recvDisplacements,
                HcclDataType::HCCL_DATA_TYPE_INT8);
            hccl.Wait(handle);
        }

        // Keep every AIV in the same lifetime even though only AIV0 may write
        // the HCCL message area and submit the collective.
        SyncAll<true>();
        hccl.Finalize();
    }
}
