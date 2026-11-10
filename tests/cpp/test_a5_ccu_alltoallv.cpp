// Production CCU graph under an eight-rank CPU scheduler. Not NPU evidence.
#include "../../examples/a5_ccu_urma_route_probe/op_kernel_ccu/alltoallv_kernel.h"
#include "../../examples/a5_ccu_urma_route_probe/common/peer_plan.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <set>
using namespace a5_ccu_urma_probe;
class Kernel : public AllToAllVKernel {
public:
    using AllToAllVKernel::AllToAllVKernel;
    using AllToAllVKernel::GeneArgs;
};
static std::vector<uint32_t> Peers(uint32_t rank)
{
    std::vector<uint32_t> peers;
    for(uint32_t d=0;d<8;++d) if(d!=rank) peers.push_back(d);
    return peers;
}
static void InvalidLayouts()
{
    std::array<uint64_t,64> counts{};
    AllToAllVLayout l;
    auto peers=Peers(0);
    assert(BuildAllToAllVLayout(0,counts.data(),64,1,peers,&l));
    assert(l.sendBytes==0 && l.recvBytes==0 && l.pathBytes.size()==7);
    assert(!BuildAllToAllVLayout(8,counts.data(),64,1,peers,&l));
    assert(!BuildAllToAllVLayout(0,nullptr,64,1,peers,&l));
    assert(!BuildAllToAllVLayout(0,counts.data(),63,1,peers,&l));
    assert(!BuildAllToAllVLayout(0,counts.data(),64,0,peers,&l));
    auto duplicate=peers; duplicate[1]=duplicate[0];
    assert(!BuildAllToAllVLayout(0,counts.data(),64,1,duplicate,&l));
    counts[0]=UINT64_MAX;
    assert(!BuildAllToAllVLayout(0,counts.data(),64,2,peers,&l));
    counts[1]=1;
    assert(!BuildAllToAllVLayout(0,counts.data(),64,1,peers,&l)); // row overflow
    counts={}; counts[1]=UINT64_MAX; counts[9]=1;
    assert(!BuildAllToAllVLayout(0,counts.data(),64,1,peers,&l)); // remote column overflow
    l.sendBytes=2; l.recvBytes=3;
    assert(DisjointAllToAllVBuffers(100,2,200,3,l));
    assert(!DisjointAllToAllVBuffers(100,1,200,3,l));
    assert(!DisjointAllToAllVBuffers(100,2,101,3,l));
    assert(!DisjointAllToAllVBuffers(UINT64_MAX,2,200,3,l));
    std::array<uint64_t,64> storage{}; storage[4]=20;
    counts={}; counts[4]=21;
    assert(!BuildAllToAllVLayoutWithStorage(0,counts.data(),64,2,peers,storage.data(),64,&l));
    counts[4]=10;
    assert(BuildAllToAllVLayoutWithStorage(0,counts.data(),64,2,peers,storage.data(),64,&l));
    assert(l.sendBytes==40 && l.pathBytes[3]==20);
    assert(!BuildAllToAllVLayoutWithStorage(0,counts.data(),64,2,peers,storage.data(),63,&l));
    std::vector<PeerPathSpec> specs;
    for(uint32_t a=0;a<8;++a) for(uint32_t b=a+1;b<8;++b) {
        PeerPathSpec p; p.srcRank=a;p.dstRank=b;p.srcPhy=a;p.dstPhy=b;specs.push_back(p);
    }
    std::string error;
    assert(ValidatePeerPlan(8,8,specs,&error));
    specs[0].relay=true; specs[0].relayPhy=9;
    assert(!ValidatePeerPlan(8,10,specs,&error));
}
int main()
{
    InvalidLayouts();
    constexpr size_t guard=37;
    std::array<std::unique_ptr<Kernel>,8> kernels;
    for(uint32_t r=0;r<8;++r) {
        std::vector<ChannelHandle> channels;
        for(auto p:Peers(r)) channels.push_back(p+1);
        AllToAllVKernelArg arg(channels); kernels[r].reset(new Kernel(arg));
        assert(kernels[r]->Init()==HCCL_SUCCESS);
    }
    size_t zeroDirections=0, asymmetricSelf=0, launches=0;
    // Bits persist across epochs: every handshake must consume its notification.
    uint32_t bits[8][8][3]{};
    for(uint32_t epoch=0;epoch<100;++epoch) {
        std::array<uint64_t,64> counts{};
        if(epoch%10==1) { for(uint32_t r=0;r<8;++r) counts[r*8+r]=r+1; }
        else if(epoch%10==2) { counts[2*8+5]=13; counts[3*8+3]=7; }
        else if(epoch%10>=3) {
            for(uint32_t s=0;s<8;++s) for(uint32_t d=0;d<8;++d)
                counts[s*8+d]=(s*17+d*31+epoch*7+s*d)%19;
            for(uint32_t d=0;d<8;++d) counts[7*8+d]=0; // empty sender, nonempty receiver
            for(uint32_t s=0;s<8;++s) counts[s*8+6]=0; // empty receiver, nonempty sender
        }
        auto transfer=counts;
        if(epoch%2) for(size_t f=0;f<64;++f) transfer[f]/=2;
        const uint64_t rowBytes=std::array<uint64_t,6>{{1,2,4,14,257,514}}[epoch%6];
        std::array<AllToAllVLayout,8> layouts;
        std::array<std::vector<unsigned char>,8> input, output, original;
        std::array<hcomm::CcuRep::Env,8> envs;
        uint64_t published[8][8][2]{};
        for(uint32_t r=0;r<8;++r) {
            if(epoch%2) assert(BuildAllToAllVLayoutWithStorage(r,transfer.data(),64,rowBytes,Peers(r),counts.data(),64,&layouts[r]));
            else assert(BuildAllToAllVLayout(r,counts.data(),64,rowBytes,Peers(r),&layouts[r]));
            const auto &l=layouts[r];
            if(l.selfBytes && l.selfSourceOffset!=l.selfDestinationOffset) ++asymmetricSelf;
            input[r].assign(2*guard+std::max<uint64_t>(1,l.sendBytes),0xb7);
            output[r].assign(2*guard+std::max<uint64_t>(1,l.recvBytes),0xc3);
            for(size_t b=0;b<l.sendBytes;++b) input[r][guard+b]=(epoch*23+r*37+b)%251;
            original[r]=input[r];
            AllToAllVTaskArg task(reinterpret_cast<uint64_t>(input[r].data()+guard),
                reinterpret_cast<uint64_t>(output[r].data()+guard),11,22,l);
            envs[r].args=kernels[r]->GeneArgs(task);
            assert(envs[r].args.size()==31);
            envs[r].remote=[&,r](ChannelHandle ch,uint32_t slot) { return published[r][ch-1][slot]; };
        }
        std::array<size_t,8> pc{};
        std::array<std::vector<bool>,8> conditions;
        std::array<std::set<int>,8> completed;
        size_t ticks=0;
        for(;;) {
            bool progress=false, allDone=true;
            for(uint32_t n=0;n<8;++n) {
                const auto r=(n+epoch)%8;
                const auto &instructions=kernels[r]->instructions;
                if(pc[r]==instructions.size()) continue;
                allDone=false;
                const auto &i=instructions[pc[r]]; const auto &e=envs[r];
                const bool active=conditions[r].empty() || conditions[r].back();
                if(i.op=="if_begin") conditions[r].push_back(active && i.len.eval(e));
                else if(i.op=="if_end") { assert(!conditions[r].empty()); conditions[r].pop_back(); }
                else if(active) {
                    if(i.op=="record") {
                        const uint32_t peer=i.channel-1;
                        assert((bits[peer][r][i.slot]&i.mask)==0);
                        bits[peer][r][i.slot]|=i.mask;
                        if(i.slot<2) published[peer][r][i.slot]=i.len.eval(e);
                        else {
                            for(size_t p=0;p<7;++p) if(layouts[r].pathBytes[p])
                                assert(completed[r].count(static_cast<int>(p)));
                        }
                    } else if(i.op=="notify_wait") {
                        auto &value=bits[r][i.channel-1][i.slot];
                        if((value&i.mask)!=i.mask) continue;
                        value&=~i.mask;
                    } else if(i.op=="event_wait") {
                        assert(completed[r].count(i.event)); // never wait for an unsubmitted zero write
                    } else if(i.op=="remote" || i.op=="group_local") {
                        const uint64_t length=i.len.eval(e);
                        if(i.op=="remote") { assert(length>0); completed[r].insert(i.event); }
                        const uint32_t dst=i.op=="remote" ? i.channel-1 : r;
                        const auto src=i.src.eval(e), target=i.dst.eval(e);
                        assert(src>=reinterpret_cast<uint64_t>(input[r].data()+guard));
                        assert(src+length<=reinterpret_cast<uint64_t>(input[r].data()+guard)+layouts[r].sendBytes);
                        assert(target>=reinterpret_cast<uint64_t>(output[dst].data()+guard));
                        assert(target+length<=reinterpret_cast<uint64_t>(output[dst].data()+guard)+layouts[dst].recvBytes);
                        if(length) std::memcpy(reinterpret_cast<void *>(target),reinterpret_cast<const void *>(src),length);
                    }
                }
                ++pc[r]; progress=true;
            }
            if(allDone) break;
            assert(progress && ++ticks<1000); // detects protocol deadlocks
        }
        // Independent reference uses counts directly, not the production offsets.
        for(uint32_t d=0;d<8;++d) {
            size_t recvOffset=0;
            for(uint32_t s=0;s<8;++s) {
                uint64_t sendOffset=0;
                for(uint32_t q=0;q<d;++q) sendOffset+=counts[s*8+q]*rowBytes;
                const auto length=transfer[s*8+d]*rowBytes;
                if(!length) ++zeroDirections;
                assert(std::memcmp(output[d].data()+guard+recvOffset,
                    input[s].data()+guard+sendOffset,length)==0);
                const auto segmentBytes=counts[s*8+d]*rowBytes;
                for(size_t b=length;b<segmentBytes;++b) assert(output[d][guard+recvOffset+b]==0xc3);
                recvOffset+=segmentBytes;
            }
            for(size_t b=0;b<output[d].size();++b)
                if(b<guard || b>=guard+layouts[d].recvBytes) assert(output[d][b]==0xc3);
            assert(input[d]==original[d] && conditions[d].empty());
            for(uint32_t s=0;s<8;++s) for(uint32_t slot=0;slot<3;++slot) assert(bits[d][s][slot]==0);
            ++launches;
        }
    }
    assert(asymmetricSelf>0 && zeroDirections>0);
    std::cout << "ALLTOALLV_CPU_MODEL_PASS launches=" << launches << " epochs=100\n";
}
