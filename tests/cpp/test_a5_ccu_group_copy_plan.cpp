#include "../../examples/a5_ccu_urma_route_probe/common/group_copy_plan.h"
#include <cassert>
#include <iostream>
#include <vector>
using namespace a5_ccu_urma_probe;
static void Check(uint64_t bytes)
{
    std::array<uint64_t,4> g; assert(BuildGroupCopyPlan(bytes,&g));
    assert(g[0]==g[1]*GROUP_BATCH && g[1]<=GROUP_MAX_ITER && g[3]<=GROUP_SLICE);
    uint64_t covered=g[0];
    if(g[2]) {
        auto repeat=(g[2]>>55)&0xff, index=(g[2]>>48)&0x7f, total=(g[2]>>41)&0x7f;
        auto lanes=repeat+total; assert(lanes<=GROUP_LANES && total>=1 && total<=2);
        assert(index==(total==2?1U:0U));
        // No overlaps/gaps: residual is first, the remaining lanes are full.
        covered+=g[3]+(total==2?repeat+1:repeat)*GROUP_SLICE;
        assert(g[3]>0);
    } else assert(g[3]==0);
    assert(covered==bytes);
}
int main()
{
    for(uint64_t m:{0ULL,1ULL,16ULL,8190ULL})
        for(uint64_t n=0;n<8;++n)
            for(uint64_t p:{0ULL,1ULL,4ULL,255ULL,256ULL,4095ULL,32767ULL}) {
                uint64_t bytes=m*GROUP_BATCH+n*GROUP_SLICE+p;
                if(bytes) Check(bytes);
            }
    for(uint64_t bytes:std::vector<uint64_t>{4096,4194304,8388608,GROUP_MAX_BYTES}) Check(bytes);
    std::array<uint64_t,4> go;
    assert(!BuildGroupCopyPlan(0,&go)); assert(!BuildGroupCopyPlan(GROUP_MAX_BYTES+1,&go));
    assert(!BuildGroupCopyPlan(4096,nullptr));
    assert(BuildGroupCopyPlan(4194304,&go)); assert(go[0]==4194304 && go[1]==16 && go[2]==0);
    std::cout<<"GROUP_PLAN_CPU_PASS: bulk/tail coverage, boundaries and 4MiB=8 lanes x16x32KiB; not hardware proof\n";
}
