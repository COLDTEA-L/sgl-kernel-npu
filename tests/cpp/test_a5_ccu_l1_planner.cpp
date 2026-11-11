#include "../../csrc/deepep/ep8_relay_planner.hpp"
#include "../../examples/a5_ccu_urma_route_probe/common/eid_alltoallv_layout.h"
#include "../../examples/a5_ccu_urma_route_probe/common/peer_plan.h"
#include <cassert>
#include <iostream>
#include <numeric>
using namespace deep_ep::ep8;
int main() {
    for(uint32_t ranks:{2u,4u,8u}) {
        std::vector<a5_ccu_urma_probe::PeerPathSpec> specs;
        for(uint32_t s=0;s<ranks;++s)for(uint32_t d=s+1;d<ranks;++d) {
            a5_ccu_urma_probe::PeerPathSpec direct;direct.srcRank=s;direct.dstRank=d;direct.srcPhy=s;direct.dstPhy=d;
            specs.push_back(direct);auto l1=direct;l1.l1=true;l1.srcEid="observed-a";l1.dstEid="observed-b";specs.push_back(l1);
        }
        std::string error;assert(a5_ccu_urma_probe::ValidatePeerPlan(ranks,8,specs,&error,ranks==8));
        specs.push_back(specs.back());assert(!a5_ccu_urma_probe::ValidatePeerPlan(ranks,8,specs,&error,ranks==8));
    }
    std::vector<std::vector<U>> allowed(64),ports(64);
    for(size_t s=0;s<8;++s) for(size_t d=0;d<8;++d) if(s!=d) {
        ports[s*8+d]={s*18+1,d*18+1};
        for(size_t k=0;k<8;++k) if(k!=s && k!=d) allowed[s*8+d].push_back(k);
    }
    auto base=Planner(allowed,5120,1,0,56,{});
    auto planner=base.WithL1Routes(ports);
    Matrix c{};c[1]=1000;
    auto p=planner.Make(c);
    assert(p.relays[1]==8 && p.moved[1]==500 && p.loads[1]==500);
    bool rejected=false;
    try{planner.Build(c,p.relays,p.moved,4096,128);}catch(const std::invalid_argument&){rejected=true;}
    assert(rejected);
    auto bad=ports;bad[1]={18,19};rejected=false;
    try{base.WithL1Routes(bad);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    bad=ports;bad[1].clear();rejected=false;
    try{base.WithL1Routes(bad);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    // Shared ingress: all seven sources use the same destination physical port.
    c.fill(0);for(size_t s=1;s<8;++s)c[s*8]=1000;
    p=planner.Make(c);assert(p.l1Rx[1]<=1000);
    assert(*std::max_element(p.loads.begin(),p.loads.end())>=875);
    // All 56 flows need splitting: the old hidden eight-action cap prevented it.
    c.fill(3072);
    p=planner.Make(c);
    assert(p.actions.size()==56 && p.resourcePeakAfter==2688);
    for(size_t f=0;f<64;++f) if(f/8!=f%8) assert(p.moved[f]==384);
    auto limited=Planner(allowed,5120,1,0,16,{}).WithL1Routes(ports).Make(c);
    assert(limited.actions.empty()); // Never publish a partial target assignment.
    std::vector<std::vector<U>> dedicated(64);
    for(size_t s=0;s<8;++s)for(size_t d=0;d<8;++d)if(s!=d)
        dedicated[s*8+d]={s*18+d,d*18+s};
    auto independent=base.WithL1Routes(dedicated);
    p=independent.Make(c);
    assert(p.actions.size()==56 && p.resourcePeakAfter==1536 && p.resourceLowerBound==1536);
    for(size_t f=0;f<64;++f) if(f/8!=f%8) assert(p.moved[f]==1536);
    std::vector<U> cap(144,1000),background(144,0),half(144,500),zero(144,0);
    auto slow=independent.WithL1Budget(half,half,background,background);
    p=slow.Make(c);assert(p.resourcePeakAfter==2048);
    // An unavailable L1 fabric does not create fake cut capacity.
    p=planner.WithL1Budget(zero,zero,background,background).Make(c);
    assert(p.actions.empty() && p.resourceLowerBound==3072);
    // Pre-existing traffic is bytes, not FP8 rows; combine must re-evaluate it.
    std::vector<std::vector<U>> noRelays(64),one(64);
    one[1]={1,19};one[8]={19,1};
    auto single=Planner(noRelays,5120,1,0,56,{}).WithL1Routes(one);
    auto txBase=background,rxBase=background;txBase[1]=1000*5120;rxBase[19]=1000*5120;
    auto busy=single.WithL1Budget(cap,cap,txBase,rxBase);
    c.fill(0);c[1]=1000;
    p=busy.Make(c);assert(p.actions.empty() && p.resourcePeakBefore==1000);
    p=busy.WithRowBytes(10240).Make(c);
    assert(p.relays[1]==8 && p.moved[1]==250 && p.resourcePeakAfter==750);
    assert(single.Make(c).moved[1]==500); // Immutable budget snapshot.
    rejected=false;try{single.WithL1Budget(zero,cap,txBase,rxBase);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    rejected=false;auto invalidCap=cap;invalidCap[0]=1001;
    try{single.WithL1Budget(invalidCap,cap,background,background);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    // Wide arithmetic must not wrap even when a controller supplies large work.
    txBase[1]=Max;cap[1]=1;
    p=single.WithL1Budget(cap,cap,txBase,rxBase).Make(c);
    assert(p.actions.empty() && p.resourcePeakBefore>=1000);
    uint64_t seed=97;
    for(unsigned iter=0;iter<1000;++iter) {
        for(auto& v:c){seed=seed*6364136223846793005ULL+1;v=(seed>>32)%3000;}
        for(U rb:{5120u,10240u}) {
            p=planner.WithRowBytes(rb).Make(c);
            Matrix rebuilt=c;for(size_t k=0;k<8;++k)rebuilt[k*8+k]=0;
            std::array<U,144> tx{},rx{};
            auto originalPeak=*std::max_element(rebuilt.begin(),rebuilt.end());
            for(size_t f=0;f<64;++f) if(p.moved[f]) {
                U x=p.moved[f];int k=p.relays[f];assert(x<=c[f]);rebuilt[f]-=x;
                if(k==8){tx[ports[f][0]]+=x;rx[ports[f][1]]+=x;}
                else{assert(k>=0 && k<8);rebuilt[f/8*8+k]+=x;rebuilt[k*8+f%8]+=x;}
            }
            assert(p.loads==rebuilt && p.l1Tx==tx && p.l1Rx==rx);
            U peak=std::max({*std::max_element(rebuilt.begin(),rebuilt.end()),*std::max_element(tx.begin(),tx.end()),*std::max_element(rx.begin(),rx.end())});
            assert(p.actions.empty()?peak==originalPeak:peak<originalPeak);
            for(uint32_t rank=0;rank<8;++rank) {
                std::vector<a5_ccu_urma_probe::EidChannel> channels;
                std::array<uint16_t,64> masks{};
                for(size_t f=0;f<64;++f)if(f/8!=f%8){masks[f]=1u<<8;for(auto k:allowed[f])masks[f]|=1u<<k;}
                for(uint32_t peer=0;peer<8;++peer)if(peer!=rank){channels.push_back({peer,-1});channels.push_back({peer,8});for(auto k:allowed[rank*8+peer])channels.push_back({peer,int(k)});}
                a5_ccu_urma_probe::EidAllToAllVLayout ref,fast;
                a5_ccu_urma_probe::EidLayoutTopology topology;
                assert(topology.Init(rank,channels));
                assert(a5_ccu_urma_probe::BuildEidAllToAllVLayout(rank,c.data(),64,rb,p.relays.data(),p.moved.data(),masks,channels,&ref));
                assert(topology.Build(c.data(),64,rb,p.relays.data(),p.moved.data(),masks,&fast));
                assert(ref.peerArgs==fast.peerArgs && ref.pathBytes==fast.pathBytes && ref.activeMask==fast.activeMask);
            }
        }
    }
    // Independent conservation/port-capacity check with fractional bandwidth,
    // nonzero competing bytes, and both dispatch/combine row widths.
    for(unsigned iter=0;iter<200;++iter) {
        for(auto& v:c){seed=seed*6364136223846793005ULL+1;v=(seed>>32)%3000;}
        for(size_t r=0;r<144;++r){cap[r]=250+(r%4)*250;txBase[r]=(r%5)*12345;rxBase[r]=(r%3)*23456;}
        for(U rb:{5120u,10240u}) {
            p=planner.WithL1Budget(cap,cap,txBase,rxBase).WithRowBytes(rb).Make(c);
            Matrix rebuilt=c;for(size_t r=0;r<8;++r)rebuilt[r*8+r]=0;
            std::array<U,144> tx{},rx{};
            for(size_t f=0;f<64;++f)if(p.moved[f]) {
                U x=p.moved[f];auto k=p.relays[f];assert(x<=c[f]);rebuilt[f]-=x;
                if(k==8){tx[ports[f][0]]+=x;rx[ports[f][1]]+=x;}
                else{assert(k>=0 && k<8);rebuilt[f/8*8+k]+=x;rebuilt[k*8+f%8]+=x;}
            }
            assert(rebuilt==p.loads && tx==p.l1Tx && rx==p.l1Rx);
            U peak=*std::max_element(rebuilt.begin(),rebuilt.end());
            for(size_t r=0;r<144;++r) {
                peak=std::max(peak,((tx[r]*rb+txBase[r])*1000+rb*cap[r]-1)/(rb*cap[r]));
                peak=std::max(peak,((rx[r]*rb+rxBase[r])*1000+rb*cap[r]-1)/(rb*cap[r]));
            }
            assert(peak==p.resourcePeakAfter && peak<=p.resourcePeakBefore);
            assert(p.actions.empty() || peak<p.resourcePeakBefore);
        }
    }
    std::cout<<"L1_PLANNER_PASS random=1000 dtypes=2 shared_port_budget=checked layouts=16000\n";
}
