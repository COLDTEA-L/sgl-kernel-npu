#ifndef HCCL_ALL2_ALL_CCU_TILING_H
#define HCCL_ALL2_ALL_CCU_TILING_H

#include "kernel_tiling/kernel_tiling.h"

struct HcclAll2AllCcuInfo {
    uint32_t rankSize;
    uint32_t rankId;
    uint64_t sendCount;
    uint64_t perRankBytes;
};

struct HcclAll2AllCcuTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    HcclAll2AllCcuInfo info;
};

#endif
