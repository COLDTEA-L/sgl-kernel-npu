#include "data_type.h"
#include <cassert>
#include <iostream>
using namespace a5_ccu_urma_probe;
int main()
{
    assert(PayloadElementBytes(HCCL_DATA_TYPE_FP16) == 2);
    assert(PayloadElementBytes(HCCL_DATA_TYPE_BFP16) == 2);
    assert(PayloadElementBytes(HCCL_DATA_TYPE_FP32) == 4);
    assert(PayloadElementBytes(HCCL_DATA_TYPE_INT8) == 1);
    assert(PayloadElementBytes(HCCL_DATA_TYPE_INT32) == 4);
    assert(PayloadElementBytes(HCCL_DATA_TYPE_RESERVED) == 0);
    assert(PayloadElementBytes(HCCL_DATA_TYPE_FP64) == 0);
    std::cout << "HCCL payload dtype/element bytes: PASS\n";
}
