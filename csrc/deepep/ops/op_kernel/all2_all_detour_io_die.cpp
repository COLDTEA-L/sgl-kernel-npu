#include <cstddef>

#include "kernel_operator.h"
#include "lib/hccl/hccl.h"
#include "common.h"
#include "all2_all_detour_io_die_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t MAX_CCU_RANKS = 32U;
constexpr uint32_t PARAM_ARRAY_COUNT = 2U;
constexpr uint32_t COPY_BUFFER_BYTES = 32U * 1024U;
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
    const uint64_t windowStrideBytes = tilingData.info.windowStrideBytes;

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
        TBuf<TPosition::VECCALC> copyBuf;
        pipe.InitBuffer(paramBuf, PARAM_ARRAY_COUNT * MAX_CCU_RANKS * sizeof(uint64_t));
        pipe.InitBuffer(copyBuf, COPY_BUFFER_BYTES);
        LocalTensor<uint64_t> params = paramBuf.Get<uint64_t>();
        LocalTensor<uint8_t> copyLocal = copyBuf.Get<uint8_t>();

        const uint32_t sendOffsetsBase = 0U;
        const uint32_t sendSizesBase = MAX_CCU_RANKS;
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
            params.SetValue(sendOffsetsBase + peer, static_cast<uint64_t>(peerCommIndex) * perRankBytes);
            params.SetValue(sendSizesBase + peer,
                            (selfParticipates && peerParticipates) ? perRankBytes : 0UL);
        }

        GlobalTensor<uint64_t> sendOffsets;
        GlobalTensor<uint64_t> sendSizes;
        sendOffsets.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(workspace));
        sendSizes.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint64_t *>(workspace + MAX_CCU_RANKS * sizeof(uint64_t)));
        DataCopyExtParams paramCopyParams = {
            1U, static_cast<uint32_t>(rankSize * sizeof(uint64_t)), 0U, 0U, 0U};
        DataCopyPad(sendOffsets, params[sendOffsetsBase], paramCopyParams);
        DataCopyPad(sendSizes, params[sendSizesBase], paramCopyParams);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);

        const uint64_t remoteWindowOffset = static_cast<uint64_t>(rankId) * windowStrideBytes;
        const uint64_t localDataSize = selfParticipates ? perRankBytes : 0UL;
        HcclHandle handle = hccl.AlltoAllvWrite<true>(
            sendData, workspace, workspace + MAX_CCU_RANKS * sizeof(uint64_t),
            remoteWindowOffset, localDataSize);
        hccl.Wait(handle);

        if (selfParticipates) {
            GlobalTensor<uint8_t> localWindow;
            GlobalTensor<uint8_t> output;
            localWindow.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(context->windowsOut[0]));
            output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(recvData));
            DataCopyPadExtParams<uint8_t> padParams = {false, 0U, 0U, 0U};

            for (uint32_t i = 0; i < commRankCount; ++i) {
                const uint32_t sourceRank = static_cast<uint32_t>(commRanks.GetValue(i));
                uint64_t bytesCopied = 0UL;
                while (bytesCopied < perRankBytes) {
                    const uint32_t copyBytes = static_cast<uint32_t>(
                        (perRankBytes - bytesCopied > COPY_BUFFER_BYTES)
                            ? COPY_BUFFER_BYTES
                            : perRankBytes - bytesCopied);
                    const uint64_t sourceOffset =
                        static_cast<uint64_t>(sourceRank) * windowStrideBytes + bytesCopied;
                    const uint64_t outputOffset = static_cast<uint64_t>(i) * perRankBytes + bytesCopied;
                    DataCopyExtParams copyParams = {1U, copyBytes, 0U, 0U, 0U};
                    DataCopyPad(copyLocal, localWindow[sourceOffset], copyParams, padParams);
                    AscendC::SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                    DataCopyPad(output[outputOffset], copyLocal, copyParams);
                    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    bytesCopied += copyBytes;
                }
            }
        }
    }

    SyncAll<true>();
    if (GetBlockIdx() == 0) {
        hccl.Finalize();
    }
}
