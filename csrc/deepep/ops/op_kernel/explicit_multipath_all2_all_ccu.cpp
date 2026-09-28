#include <cstddef>

#include "kernel_operator.h"
#include "lib/hccl/hccl.h"
#include "common.h"
#include "explicit_multipath_all2all_ccu_tiling.h"

using namespace AscendC;

// AIV is control-only. Payload movement is performed by the CCU kernel and
// URMA channels provisioned for plan_id by the HCCL extension.
extern "C" __global__ __aicore__ void explicit_multipath_all2_all_ccu(
    GM_ADDR sendData, GM_ADDR pathPolicy, GM_ADDR recvData,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ExplicitMultipathAll2AllCcuTilingData);
    GET_TILING_DATA_WITH_STRUCT(ExplicitMultipathAll2AllCcuTilingData, tilingData, tiling);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ HcclCombineOpParam *context =
        reinterpret_cast<__gm__ HcclCombineOpParam *>(GetHcclContext<0>());
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(context), &tilingData);
    hccl.SetCcTilingV2(offsetof(ExplicitMultipathAll2AllCcuTilingData, mc2CcTiling));

    if ASCEND_IS_AIV {
        SyncAll<true>();
        if (GetBlockIdx() == 0) {
            GlobalTensor<int64_t> policy;
            policy.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(pathPolicy), 1);
            // Explicit multipath repurposes strideCount as a runtime policy
            // carrier.  The paired HCCL template decodes it instead of using
            // native AllToAll stride semantics.
            const uint64_t policyWord = static_cast<uint64_t>(policy.GetValue(0));
            HcclHandle handle = hccl.AlltoAll<true>(
                sendData, recvData, tilingData.info.perRankBytes,
                HcclDataType::HCCL_DATA_TYPE_INT8, policyWord, 1);
            hccl.Wait(handle);
        }
        SyncAll<true>();
        hccl.Finalize();
    }
}
