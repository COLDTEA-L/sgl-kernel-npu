#ifndef CCU_HBM_COMMAND_PUNCTURE_TILING_H
#define CCU_HBM_COMMAND_PUNCTURE_TILING_H

#include "kernel_tiling/kernel_tiling.h"

constexpr uint32_t A5_COMMAND_PUNCTURE_MAX_PATHS = 8U;

struct CcuHbmCommandPunctureInfo {
    uint32_t rankId;
    uint32_t pathCount;
    uint32_t transfer;
    uint32_t reserved;
    uint64_t perPeerBytes;
    uint64_t maxSpinCount;
    uint32_t pathWeights[A5_COMMAND_PUNCTURE_MAX_PATHS];
};

struct CcuHbmCommandPunctureTilingData {
    CcuHbmCommandPunctureInfo info;
};

#endif
