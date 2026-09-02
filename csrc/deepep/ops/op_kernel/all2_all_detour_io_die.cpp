#include <cstddef>

#include "kernel_operator.h"
#include "lib/hccl/hccl.h"
#include "common.h"
#include "all2_all_detour_io_die_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t DUAL_DIE_COUNT = 2U;
constexpr uint32_t COPY_BUFFER_BYTES = 32U * 1024U;
}

extern "C" __global__ __aicore__ void all2_all_detour_io_die(
    GM_ADDR sendData, GM_ADDR commRankIds, GM_ADDR recvData, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(All2AllDetourIoDieTilingData);
    GET_TILING_DATA_WITH_STRUCT(All2AllDetourIoDieTilingData, tilingData, tiling);

    const uint32_t commRankCount = tilingData.info.commRankCount;
    const uint64_t perRankBytes = tilingData.info.perRankBytes;
    const uint64_t windowStrideBytes = tilingData.info.windowStrideBytes;

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

        GM_ADDR sendSizesGM = workspace;
        GM_ADDR sendOffsetsGM = workspace + DUAL_DIE_COUNT * rankSize * sizeof(uint64_t);
        TPipe pipe;
        TBuf<TPosition::VECCALC> copyBuf;
        pipe.InitBuffer(copyBuf, COPY_BUFFER_BYTES);
        LocalTensor<uint8_t> copyLocal = copyBuf.Get<uint8_t>();

        // AlltoAllvWrite's CCU implementation enters on every AIV. Follow the
        // CANN sample exactly here: every AIV writes the same two-die arrays
        // directly to GM before entering the collective. Restricting this to
        // block 0 and adding a pre-collective SyncAll changes the CCU task
        // sequence and can make the launch fail with ACLNN status 561000.
        __gm__ uint64_t *sendSizes = reinterpret_cast<__gm__ uint64_t *>(sendSizesGM);
        __gm__ uint64_t *sendOffsets = reinterpret_cast<__gm__ uint64_t *>(sendOffsetsGM);
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
            const uint64_t peerOffset = static_cast<uint64_t>(peerCommIndex) * perRankBytes;
            const uint64_t die0Bytes = perRankBytes / DUAL_DIE_COUNT;
            const uint64_t die1Bytes = perRankBytes - die0Bytes;
            const bool shouldSend = selfParticipates && peerParticipates;
            sendSizes[peer] = shouldSend ? die0Bytes : 0UL;
            sendSizes[rankSize + peer] = shouldSend ? die1Bytes : 0UL;
            sendOffsets[peer] = peerOffset;
            sendOffsets[rankSize + peer] = peerOffset + die0Bytes;
        }

        const uint64_t remoteWindowOffset = static_cast<uint64_t>(rankId) * windowStrideBytes;
        const uint64_t localDataSize = selfParticipates ? perRankBytes : 0UL;
        HcclHandle handle = hccl.AlltoAllvWrite<true>(
            sendData, sendOffsetsGM, sendSizesGM, remoteWindowOffset, localDataSize);
        hccl.Wait(handle);
        SyncAll<true>();

        if (GetBlockIdx() == 0 && selfParticipates) {
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

        // All AIV cores must finish consuming the local window before any of
        // them tears down the CCU task state.
        SyncAll<true>();
        hccl.Finalize();
    }
}
