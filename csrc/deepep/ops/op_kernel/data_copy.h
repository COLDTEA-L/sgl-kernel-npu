#ifndef CAM_DATACOPY_GM2GM_H
#define CAM_DATACOPY_GM2GM_H
#include <type_traits>
#include "comm_args.h"

using namespace AscendC;
using namespace Moe;

template <typename T>
FORCE_INLINE_AICORE void SetAtomicOpType(int op)
{
    switch (op) {
        case ADD:
            AscendC::SetAtomicAdd<T>();
            break;
        case MUL:
            // Ignore setting the atomic register when performing mul
            break;
        case MAX:
            AscendC::SetAtomicMax<T>();
            break;
        case MIN:
            AscendC::SetAtomicMin<T>();
            break;
        default:
            AscendC::SetAtomicNone();
    }
}

template <typename T>
FORCE_INLINE_AICORE void CpUB2GM(__gm__ T *gmAddr, __ubuf__ T *ubAddr, uint32_t size)
{
    LocalTensor<uint8_t> ubTensor;
    GlobalTensor<uint8_t> gmTensor;
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
    DataCopyPad(gmTensor, ubTensor, dataCopyParams);
}

template <typename T>
FORCE_INLINE_AICORE void CpGM2UB(__ubuf__ T *ubAddr, __gm__ T *gmAddr, uint32_t size)
{
    LocalTensor<uint8_t> ubTensor;
    GlobalTensor<uint8_t> gmTensor;
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
    DataCopyPadExtParams<uint8_t> padParams;
    DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
}

/**
 * Copy contiguous data from GM to GM through two alternating UB buffers.
 *
 * count is expressed in elements of T.  The two MTE pipelines can overlap the
 * GM->UB load for one chunk with the UB->GM store for the previous chunk.  The
 * first 96 bytes of UB remain available to callers for scalar communication
 * flags; the two data buffers use the same layout as NotifyDispatchA5.
 */
template <typename T>
FORCE_INLINE_AICORE void CpGM2GM(__gm__ T *dst, __gm__ T *src, uint64_t count)
{
    if (count == 0UL) {
        return;
    }

    constexpr uint32_t maxCountPerLoop =
        static_cast<uint32_t>(UB_SINGLE_PING_PONG_ADD_SIZE_MAX / sizeof(T));
    __ubuf__ T *copyUb[2] = {
        (__ubuf__ T *)(UB_HEAD_OFFSET),
        (__ubuf__ T *)(UB_MID_OFFSET),
    };

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    uint64_t offset = 0UL;
    uint32_t turn = 0U;
    while (offset < count) {
        const uint64_t remaining = count - offset;
        const uint32_t currentCount = static_cast<uint32_t>(
            remaining > maxCountPerLoop ? maxCountPerLoop : remaining);
        const event_t eventId = turn == 0U ? EVENT_ID0 : EVENT_ID1;

        // Do not reuse this UB half until its previous MTE3 store completes.
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        CpGM2UB(copyUb[turn], src + offset, currentCount * sizeof(T));
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        CpUB2GM(dst + offset, copyUb[turn], currentCount * sizeof(T));
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);

        offset += currentCount;
        turn ^= 1U;
    }

    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
    AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
}

template <typename T>
FORCE_INLINE_AICORE void CopyUB2UB(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t calCount)
{
    LocalTensor<T> srcTensor;
    LocalTensor<T> dstTensor;
    TBuffAddr srcAddr, dstAddr;
    srcAddr.bufferAddr = reinterpret_cast<uint64_t>(src);
    dstAddr.bufferAddr = reinterpret_cast<uint64_t>(dst);
    srcTensor.SetAddr(srcAddr);
    dstTensor.SetAddr(dstAddr);
    DataCopy(dstTensor, srcTensor, calCount);
}

#endif  // CAM_DATACOPY_GM2GM_H
