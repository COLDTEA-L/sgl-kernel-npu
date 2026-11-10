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
    std::cout<<"L1_PLANNER_PASS random=1000 dtypes=2 shared_port_budget=checked layouts=16000\n";
}
