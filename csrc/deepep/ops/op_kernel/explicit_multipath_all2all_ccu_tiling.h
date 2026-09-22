#ifndef EXPLICIT_MULTIPATH_ALL2ALL_CCU_TILING_H
#define EXPLICIT_MULTIPATH_ALL2ALL_CCU_TILING_H

#include "kernel_tiling/kernel_tiling.h"

constexpr uint32_t A5_EXPLICIT_MULTIPATH_MAX_PATHS = 8U;

struct ExplicitMultipathAll2AllCcuInfo {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t pathCount;
    uint32_t reserved;
    uint64_t sendCount;
    uint64_t perRankBytes;
    uint64_t planHash;
    uint32_t pathWeights[A5_EXPLICIT_MULTIPATH_MAX_PATHS];
};

struct ExplicitMultipathAll2AllCcuTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    ExplicitMultipathAll2AllCcuInfo info;
};

#endif
