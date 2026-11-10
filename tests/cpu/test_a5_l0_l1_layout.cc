#include "../../examples/a5_ccu_l0_l1_probe/layout.h"
#include <cassert>
#include <iostream>
int main() {
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
