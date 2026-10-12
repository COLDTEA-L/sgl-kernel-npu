#include "../../examples/a5_ccu_urma_route_probe/common/group_copy_plan.h"
#include <cassert>
#include <iostream>
namespace ops_hccl {
uint64_t GetLoopParam(uint64_t, uint64_t, uint64_t);
uint64_t GetParallelParam(uint64_t, uint64_t, uint64_t);
uint64_t GetOffsetParam(uint64_t, uint64_t, uint64_t);
}
int main()
{
    using namespace a5_ccu_urma_probe;
    for (uint64_t step : {0ULL, 32768ULL, 262144ULL})
        for (uint64_t n : {0ULL, 1ULL, 16ULL, 8191ULL})
            assert(GroupLoopParam(step,n)==ops_hccl::GetLoopParam(0,step,n));
    for (uint64_t r=0;r<8;++r)
        for (uint64_t total=1;total<=2;++total)
            assert(GroupParallelParam(r,total-1,total)==ops_hccl::GetParallelParam(r,total-1,total));
    assert(GroupOffsetParam()==ops_hccl::GetOffsetParam(GROUP_SLICE,GROUP_MS_STRIDE,1));
    std::cout << "NATIVE_GROUP_CODEC_PASS (old-object SDK; no NPU)\n";
}
