#ifndef EXPLICIT_MULTIPATH_ALL2ALL_CCU_TILING_H
#define EXPLICIT_MULTIPATH_ALL2ALL_CCU_TILING_H

#include "kernel_tiling/kernel_tiling.h"

// The CCU launch ABI has room for 48 uint64 task arguments.  Six fixed
// arguments, three arguments per path and three cache metadata words allow
// at most thirteen pre-provisioned paths in one launch.
constexpr uint32_t A5_EXPLICIT_MULTIPATH_MAX_PATHS = 13U;

struct ExplicitMultipathAll2AllCcuInfo {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t maxPaths;
    uint32_t policyAbi;
    uint64_t sendCount;
    uint64_t perRankBytes;
    uint64_t planHash;
};

struct ExplicitMultipathAll2AllCcuTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    ExplicitMultipathAll2AllCcuInfo info;
};

#endif
