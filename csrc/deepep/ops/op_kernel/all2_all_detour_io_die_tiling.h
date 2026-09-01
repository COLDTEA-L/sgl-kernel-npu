#ifndef ALL2_ALL_DETOUR_IO_DIE_TILING_H
#define ALL2_ALL_DETOUR_IO_DIE_TILING_H

#include "kernel_tiling/kernel_tiling.h"

struct All2AllDetourIoDieInfo {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t commRankCount;
    uint32_t reserved;
    uint64_t sendCount;
    uint64_t perRankBytes;
    uint64_t windowStrideBytes;
};

struct All2AllDetourIoDieTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    All2AllDetourIoDieInfo info;
};

#endif
