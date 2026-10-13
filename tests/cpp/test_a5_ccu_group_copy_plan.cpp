#include "../../examples/a5_ccu_urma_route_probe/common/group_copy_plan.h"
#include <cassert>
#include <iostream>
#include <vector>
using namespace a5_ccu_urma_probe;
static void Check(uint64_t bytes,const GroupCopyConfig &config=GroupCopyConfig())
{
    std::array<uint64_t,4> g; assert(BuildGroupCopyPlan(bytes,&g,config));
    assert(g[0]==g[1]*config.BatchBytes() && g[1]<=GROUP_MAX_ITER && g[3]<=config.sliceBytes);
    uint64_t covered=g[0];
    if(g[2]) {
        auto repeat=(g[2]>>55)&0xff, index=(g[2]>>48)&0x7f, total=(g[2]>>41)&0x7f;
        auto lanes=repeat+total; assert(lanes<=config.lanes && total>=1 && total<=2);
        assert(index==(total==2?1U:0U));
        // No overlaps/gaps: residual is first, the remaining lanes are full.
        covered+=g[3]+(total==2?repeat+1:repeat)*config.sliceBytes;
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
    for(uint64_t slice:{32768ULL,65536ULL,131072ULL}) {
        GroupCopyConfig config(slice);
        assert(config.Valid() && config.MsStride()*GROUP_MS_BYTES==slice);
        assert(config.MsStride()*config.lanes==64 && config.BatchBytes()==GROUP_BATCH);
        assert(GroupOffsetParam(slice)==(slice<<21)+(config.MsStride()<<10)+1);
        for(uint64_t m:{0ULL,1ULL,4ULL,16ULL})
            for(uint64_t n=0;n<config.lanes;++n)
                for(uint64_t p:std::vector<uint64_t>{0,1,256,slice-256,slice-1}) {
                    uint64_t bytes=m*config.BatchBytes()+n*slice+p;
                    if(bytes) Check(bytes,config);
                }
        Check(GROUP_MAX_BYTES,config);
        assert(BuildGroupCopyPlan(4194304,&go,config));
        assert(go[0]==4194304 && go[1]==4194304/config.BatchBytes() && go[2]==0);
    }
    for(uint64_t bad:{0ULL,16384ULL,65535ULL,262144ULL})
        assert(!BuildGroupCopyPlan(4194304,&go,GroupCopyConfig(bad)));
    assert(GroupCopyConfig(65536).lanes==4 && GroupCopyConfig(131072).lanes==2);
    std::cout<<"GROUP_PLAN_CPU_PASS: fixed64 MS, 8x32/4x64/2x128 KiB, bulk/tail coverage; not hardware proof\n";
}
