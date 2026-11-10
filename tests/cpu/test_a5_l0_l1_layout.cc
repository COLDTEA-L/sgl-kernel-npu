#include "../../examples/a5_ccu_l0_l1_probe/layout.h"
#include "../../examples/a5_ccu_l0_l1_probe/endpoint_attrs.h"
#include <cassert>
#include <iostream>
int main() {
    uint8_t eid[16]={};
    assert(!l0l1::UbEidAddress(true,true,eid));
    eid[15]=1;
    assert(l0l1::UbEidAddress(true,true,eid));
    assert(!l0l1::UbEidAddress(false,true,eid));
    assert(!l0l1::UbEidAddress(true,false,eid));
    l0l1::EndpointTable table;
    l0l1::AddEndpoint(table,"peer_eid",{0,1,0,1});
    l0l1::AddEndpoint(table,"peer_eid",{0,1,0,1}); // Same endpoint reused by multiple links.
    auto peer=l0l1::DecodeEndpoints(l0l1::EncodeEndpoints(1,table),1);
    auto a=l0l1::PeerEndpoint(peer,"peer_eid",{6,0,6,0});
    assert(a.die==1 && a.dieStatus==0 && a.bw==1 && a.bwStatus==0);
    assert(l0l1::SinglePortPair(0,1,a.bwStatus,a.bw));
    bool failed=false;
    try { l0l1::PeerEndpoint(peer,"peer_eid",{0,0,0,1}); } catch(const std::runtime_error&) {failed=true;}
    assert(failed); // Successful graph and peer reports disagree about die.
    failed=false;
    try { l0l1::PeerEndpoint(peer,"missing",{6,0,6,0}); } catch(const std::runtime_error&) {failed=true;}
    assert(failed);
    failed=false;
    try { l0l1::AddEndpoint(table,"peer_eid",{0,1,0,2}); } catch(const std::runtime_error&) {failed=true;}
    assert(failed);
    for(const auto& text : {std::string("bad"),
            std::string("L0L1_ENDPOINT_ATTRS_V1 0 0"),
            std::string("L0L1_ENDPOINT_ATTRS_V1 1 1 ep 0 3 0 1"),
            std::string("L0L1_ENDPOINT_ATTRS_V1 1 2 ep 0 1 0 1 ep 0 1 0 1"),
            std::string("L0L1_ENDPOINT_ATTRS_V1 1 0 extra")}) {
        failed=false;
        try { l0l1::DecodeEndpoints(text,1); } catch(const std::runtime_error&) {failed=true;}
        assert(failed);
    }
    l0l1::EndpointTable unknown;
    l0l1::AddEndpoint(unknown,"explicit_eid",{6,1,6,0}); // Die declared by explicit configuration, BW unverified.
    auto u=l0l1::PeerEndpoint(l0l1::DecodeEndpoints(l0l1::EncodeEndpoints(1,unknown),1),"explicit_eid",{6,1,6,0});
    assert(u.dieStatus==6 && u.die==1 && !l0l1::SinglePortPair(0,1,u.bwStatus,u.bw));
    assert(l0l1::DecodeEndpoints(l0l1::EncodeEndpoints(1,{}),1).empty());
    assert(l0l1::SinglePortPair(0,1,0,1));
    assert(!l0l1::SinglePortPair(0,2,0,2));
    assert(!l0l1::SinglePortPair(0,1,0,6));
    assert(!l0l1::SinglePortPair(0,0,0,0));
    assert(!l0l1::SinglePortPair(6,1,0,1));
    assert(!l0l1::SinglePortPair(0,1,6,1));
    for(uint64_t size: {512ULL,768ULL,65536ULL,8388608ULL,268435456ULL}) {
        for(unsigned share: {25U,50U,75U}) {
            if(size*share/100<256) continue;
            const auto p=l0l1::Split(size,share);
            assert(p[0].offset==0 && p[1].offset==p[0].bytes);
            assert(p[0].bytes+p[1].bytes==size && p[0].bytes%256==0 && p[1].bytes%256==0);
            assert(p[0].bytes && p[1].bytes);
        }
    }
    for(uint64_t size: {0ULL,256ULL,513ULL,268435712ULL}) {
        bool rejected=false; try { l0l1::Split(size,50); } catch(const std::runtime_error&) {rejected=true;}
        assert(rejected);
    }
    assert(l0l1::PairKey(0,"aaa","bbb")==l0l1::PairKey(1,"bbb","aaa"));
    assert(l0l1::Pattern(0,17,0)!=l0l1::Pattern(1,17,0));
    assert(l0l1::Pattern(0,17,0)!=l0l1::Pattern(0,31,0));
    assert(l0l1::Pattern(0,17,0)!=l0l1::Pattern(0,17,1));
    std::cout<<"L0L1_LAYOUT_CPU_PASS\n";
}
