#include <cstddef>

#include "kernel_operator.h"
#include "lib/hccl/hccl.h"
#include "common.h"
#include "hccl_all2_all_ccu_tiling.h"

using namespace AscendC;

// Thin MC2 wrapper around the fixed-count HCCL AllToAll.  The collective
// itself is selected and executed by the HCCL CCU runtime (Mesh1D/2Die on A5);
// this kernel intentionally contains no custom data movement.
extern "C" __global__ __aicore__ void hccl_all2_all_ccu(
    GM_ADDR sendData, GM_ADDR recvData, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HcclAll2AllCcuTilingData);
    GET_TILING_DATA_WITH_STRUCT(HcclAll2AllCcuTilingData, tilingData, tiling);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ HcclCombineOpParam *context =
        reinterpret_cast<__gm__ HcclCombineOpParam *>(GetHcclContext<0>());
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(context), &tilingData);
    hccl.SetCcTilingV2(offsetof(HcclAll2AllCcuTilingData, mc2CcTiling));

    if ASCEND_IS_AIV {
        SyncAll<true>();
        if (GetBlockIdx() == 0) {
            // INT8 makes dataCount equal to bytes per peer and keeps this
            // validation wrapper independent of the input tensor dtype.
            HcclHandle handle = hccl.AlltoAll<true>(
                sendData,
                recvData,
                tilingData.info.perRankBytes,
                HcclDataType::HCCL_DATA_TYPE_INT8,
                0,
                1);
            hccl.Wait(handle);
        }
        SyncAll<true>();
        hccl.Finalize();
    }
}
