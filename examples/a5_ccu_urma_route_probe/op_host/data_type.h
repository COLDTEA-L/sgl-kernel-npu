#ifndef A5_CCU_PAYLOAD_DATA_TYPE_H
#define A5_CCU_PAYLOAD_DATA_TYPE_H
#include <hccl/hccl_types.h>
#include <cstdint>
namespace a5_ccu_urma_probe {
// Only a byte-size decision: no conversion and no dtype-specific CCU kernel.
inline uint32_t PayloadElementBytes(HcclDataType type)
{
    switch (type) {
        case HCCL_DATA_TYPE_INT8: return 1;
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_BFP16: return 2;
        case HCCL_DATA_TYPE_FP32:
        case HCCL_DATA_TYPE_INT32: return 4;
        default: return 0;
    }
}
} // namespace a5_ccu_urma_probe
#endif
