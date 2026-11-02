// Differential layout checks against the unchanged allocating reference builder.
#include "../../examples/a5_ccu_urma_route_probe/common/eid_alltoallv_layout.h"
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <new>
static bool forbidAllocation=false;
void* operator new(std::size_t n) {
    if(forbidAllocation) std::abort();
    if(void* p=std::malloc(n?n:1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
using namespace a5_ccu_urma_probe;
static void Same(const EidAllToAllVLayout& a,const EidAllToAllVLayout& b) {
    assert(a.sendBytes==b.sendBytes && a.recvBytes==b.recvBytes && a.selfBytes==b.selfBytes);
    assert(a.selfSourceOffset==b.selfSourceOffset && a.selfDestinationOffset==b.selfDestinationOffset);
    assert(a.sourceOffsets==b.sourceOffsets && a.remoteOffsets==b.remoteOffsets && a.pathBytes==b.pathBytes);
    assert(a.peerArgs==b.peerArgs && a.activeMask==b.activeMask);
}
int main() {
    unsigned checked=0;
    for(unsigned limit:{0u,1u,6u}) for(uint32_t rank=0;rank<8;++rank) {
        std::vector<EidChannel> channels;
        std::array<uint8_t,64> allowed{};
        for(uint32_t s=0;s<8;++s) for(uint32_t d=0;d<8;++d) if(s!=d) {
            unsigned used=0;
            for(uint32_t k=0;k<8 && used<limit;++k) if(k!=s&&k!=d) {allowed[s*8+d]|=1u<<k;++used;}
        }
        for(uint32_t p=0;p<8;++p) if(p!=rank) {
            channels.push_back({p,-1});
            for(int k=0;k<8;++k) if(allowed[rank*8+p]&(1u<<k))channels.push_back({p,k});
        }
        // Also test channel order differing from ascending peer order.
        std::reverse(channels.begin(),channels.end());
        EidLayoutTopology topology;assert(topology.Init(rank,channels));
        EidAllToAllVLayout cached,reference;
        cached.sourceOffsets.resize(channels.size());cached.remoteOffsets.resize(channels.size());cached.pathBytes.resize(channels.size());
        const auto* sp=cached.sourceOffsets.data();const auto* rp=cached.remoteOffsets.data();const auto* bp=cached.pathBytes.data();
        for(unsigned epoch=0;epoch<200;++epoch) for(uint64_t rb:{5120u,10240u}) {
            std::array<uint64_t,64> c{},m{};std::array<int32_t,64> relay;relay.fill(-1);
            for(size_t f=0;f<64;++f) {
                c[f]=epoch%7==0?0:(epoch*19+f*31)%79;
                if(epoch%7==1 && f/8!=f%8)c[f]=0;
                for(int k=0;k<8;++k) if(c[f] && allowed[f]&(1u<<k)) {
                    m[f]=epoch%3==0?c[f]:c[f]/2;
                    if(m[f])relay[f]=k;
                    break;
                }
            }
            // Exercise changed counts/policy, receive-only ranks, zero and self.
            if(epoch%7==2)for(size_t f=rank*8;f<rank*8+8;++f){c[f]=m[f]=0;relay[f]=-1;}
            if(epoch%7==3)for(size_t f=rank;f<64;f+=8){c[f]=m[f]=0;relay[f]=-1;}
            assert(BuildEidAllToAllVLayout(rank,c.data(),64,rb,relay.data(),m.data(),allowed,channels,&reference));
            forbidAllocation=true;
            const bool ok=topology.Build(c.data(),64,rb,relay.data(),m.data(),allowed,&cached);
            forbidAllocation=false;
            assert(ok);Same(cached,reference);++checked;
            assert(sp==cached.sourceOffsets.data() && rp==cached.remoteOffsets.data() && bp==cached.pathBytes.data());
        }
        std::array<uint64_t,64> c{},m{};std::array<int32_t,64> relay;relay.fill(-1);
        auto compare=[&](uint32_t count,uint64_t rb) {
            const bool expected=BuildEidAllToAllVLayout(rank,c.data(),count,rb,relay.data(),m.data(),allowed,channels,&reference);
            assert(topology.Build(c.data(),count,rb,relay.data(),m.data(),allowed,&cached)==expected);
        };
        compare(63,5120);compare(64,0);
        c[0]=UINT64_MAX;compare(64,5120);
        c[0]=UINT64_MAX;c[1]=1;compare(64,1); // row overflow
        c.fill(0);c[0]=UINT64_MAX;c[8]=1;compare(64,1); // column overflow
        c.fill(0);c[rank*8+rank]=1;m[rank*8+rank]=1;relay[rank*8+rank]=2;compare(64,5120);
        m.fill(0);relay.fill(-1);relay[0]=8;compare(64,5120);
        relay.fill(-1);c.fill(2);m[0]=3;compare(64,5120);
        // Candidate mask advertises a relay absent from the bound channels.
        if(limit==0) {
            const unsigned peer=(rank+1)%8,k=(rank+2)%8;
            allowed[peer*8+rank]|=1u<<k;
            m.fill(0);m[peer*8+rank]=1;relay.fill(-1);relay[peer*8+rank]=k;compare(64,5120);
        }
        auto bad=channels;bad.push_back(channels[0]);assert(!topology.Init(rank,bad));
        assert(!topology.Build(c.data(),64,5120,relay.data(),m.data(),allowed,&cached));
    }
    std::printf("EP8_EID_SUBMIT_LAYOUT_PASS cases=%u row_bytes=5120,10240 allocations_after_bind=0\n",checked);
}
