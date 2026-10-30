// Execute the actual EID CCU instruction graph. Hardware EID forwarding is NOT emulated.
#include "../../examples/a5_ccu_urma_route_probe/op_kernel_ccu/eid_alltoallv_kernel.h"
#include "../../examples/a5_ccu_urma_route_probe/common/peer_plan.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <set>
using namespace a5_ccu_urma_probe;
struct Kernel:EidAllToAllVKernel {using EidAllToAllVKernel::EidAllToAllVKernel;using EidAllToAllVKernel::GeneArgs;};
static std::vector<EidChannel> Catalog(uint32_t rank) {
    std::vector<EidChannel> v;
    for(uint32_t p=0;p<8;++p) if(p!=rank) {v.push_back({p,-1});for(int k=0;k<8;++k) if(k!=int(rank)&&k!=int(p)) v.push_back({p,k});}
    return v;
}
static void Invalid() {
    std::vector<PeerPathSpec> specs;
    for(uint32_t s=0;s<8;++s) for(uint32_t d=s+1;d<8;++d) {
        PeerPathSpec p;p.srcRank=s;p.dstRank=d;p.srcPhy=7-s;p.dstPhy=7-d;p.srcEid="a";p.dstEid="b";
        specs.push_back(p);
        for(uint32_t k=0;k<8;++k) if(k!=s&&k!=d) {auto q=p;q.relay=true;q.relayPhy=7-k;specs.push_back(q);}
    }
    std::string error;
    assert(ValidatePeerPlan(8,8,specs,&error,true));
    assert(!ValidatePeerPlan(8,8,specs,&error)); // original direct-only ABI remains strict
    assert(!ValidatePeerPlan(4,8,specs,&error,true));
    auto bad=specs;bad[1].relayPhy=8;assert(!ValidatePeerPlan(8,8,bad,&error,true));
    bad=specs;bad[1].relayPhy=bad[1].srcPhy;assert(!ValidatePeerPlan(8,8,bad,&error,true));
    auto paths=Catalog(0);paths.push_back(paths[0]);assert(!ValidateEidChannels(0,paths));
    std::array<uint64_t,64> c{},m{};std::array<int32_t,64> k;k.fill(-1);std::array<uint8_t,64> allowed{};
    EidAllToAllVLayout out;
    assert(BuildEidAllToAllVLayout(0,c.data(),64,5120,k.data(),m.data(),allowed,Catalog(0),&out));
    c[1]=4;m[1]=2;k[1]=2;
    assert(!BuildEidAllToAllVLayout(0,c.data(),64,5120,k.data(),m.data(),allowed,Catalog(0),&out));
    allowed[1]=1u<<2;
    assert(BuildEidAllToAllVLayout(0,c.data(),64,5120,k.data(),m.data(),allowed,Catalog(0),&out));
    m[1]=5;assert(!BuildEidAllToAllVLayout(0,c.data(),64,5120,k.data(),m.data(),allowed,Catalog(0),&out));
    m[1]=2;c[1]=UINT64_MAX;assert(!BuildEidAllToAllVLayout(0,c.data(),64,5120,k.data(),m.data(),allowed,Catalog(0),&out));
}
int main() {
    Invalid();
    std::array<std::unique_ptr<Kernel>,8> kernels;
    std::array<uint8_t,64> allowed{};
    for(int s=0;s<8;++s) for(int d=0;d<8;++d) if(s!=d) for(int k=0;k<8;++k) if(k!=s&&k!=d) allowed[s*8+d]|=1u<<k;
    for(uint32_t r=0;r<8;++r) {
        auto paths=Catalog(r);std::vector<ChannelHandle> channels;
        for(auto p:paths) channels.push_back(p.peer*9+p.relay+2);
        EidAllToAllVKernelArg arg(channels,r,paths);kernels[r].reset(new Kernel(arg));assert(kernels[r]->Init()==0);
    }
    uint32_t bits[8][8][9][3]{};
    constexpr size_t guard=37;
    size_t activeRelay=0, skipped=0;
    for(unsigned epoch=0;epoch<100;++epoch) {
        std::array<uint64_t,64> c{},m{};std::array<int32_t,64> k;k.fill(-1);
        if(epoch%10==1) for(unsigned r=0;r<8;++r) c[r*8+r]=r+1;
        else if(epoch%10>=2) for(unsigned s=0;s<8;++s) for(unsigned d=0;d<8;++d) {
            size_t f=s*8+d;c[f]=(s*17+d*31+epoch*7)%9;
            if(s==7 || d==6) c[f]=0;
            if(c[f]&&s!=d&&(s+d+epoch)%3) {
                int relay=(s+3*d+epoch)%8;while(relay==int(s)||relay==int(d)) relay=(relay+1)%8;
                m[f]=epoch%4?c[f]/2:c[f];if(m[f]) k[f]=relay;
            }
        }
        const uint64_t rb=std::array<uint64_t,5>{{1,2,257,5120,10240}}[epoch%5];
        std::array<EidAllToAllVLayout,8> layouts;
        std::array<std::vector<unsigned char>,8> input,output,original;
        std::array<hcomm::CcuRep::Env,8> envs;
        uint64_t published[8][8][9][2]{};
        for(uint32_t r=0;r<8;++r) {
            assert(BuildEidAllToAllVLayout(r,c.data(),64,rb,k.data(),m.data(),allowed,Catalog(r),&layouts[r]));
            auto& l=layouts[r];input[r].assign(2*guard+std::max<uint64_t>(1,l.sendBytes),0xb7);
            output[r].assign(2*guard+std::max<uint64_t>(1,l.recvBytes),0xc3);
            for(size_t b=0;b<l.sendBytes;++b) input[r][guard+b]=(epoch*23+r*37+b)%251;
            original[r]=input[r];
            EidAllToAllVTaskArg task(reinterpret_cast<uint64_t>(input[r].data()+guard),reinterpret_cast<uint64_t>(output[r].data()+guard),11,22,l);
            envs[r].args=kernels[r]->GeneArgs(task);assert(envs[r].args.size()==46);
            envs[r].remote=[&,r](ChannelHandle ch,uint32_t slot){return published[r][(ch-1)/9][(ch-1)%9][slot];};
            for(size_t i=0;i<49;++i) if(!(l.activeMask&(uint64_t{1}<<i))) ++skipped;
        }
        std::array<size_t,8> pc{};std::array<std::vector<bool>,8> conditions;std::array<std::set<int>,8> completed;
        size_t ticks=0;
        for(;;) {
            bool progress=false,done=true;
            for(uint32_t n=0;n<8;++n) {
                auto r=(n+epoch)%8;const auto& code=kernels[r]->instructions;
                if(pc[r]==code.size()) continue;
                done=false;const auto& i=code[pc[r]];const auto& e=envs[r];
                bool active=conditions[r].empty()||conditions[r].back();
                if(i.op=="if_begin") conditions[r].push_back(active&&i.len.eval(e));
                else if(i.op=="if_end") {assert(!conditions[r].empty());conditions[r].pop_back();}
                else if(active) {
                    const uint32_t peer=(i.channel-1)/9,path=(i.channel-1)%9;
                    if(i.op=="record") {
                        assert(!(bits[peer][r][path][i.slot]&i.mask));bits[peer][r][path][i.slot]|=i.mask;
                        if(i.slot<2) published[peer][r][path][i.slot]=i.len.eval(e);
                        else for(size_t q=0;q<49;++q) if(layouts[r].pathBytes[q]) assert(completed[r].count(q));
                    } else if(i.op=="notify_wait") {
                        auto& b=bits[r][peer][path][i.slot];if((b&i.mask)!=i.mask) continue;b&=~i.mask;
                    } else if(i.op=="event_wait") assert(completed[r].count(i.event));
                    else if(i.op=="remote"||i.op=="group_local") {
                        auto len=i.len.eval(e);uint32_t dst=i.op=="remote"?peer:r;
                        if(i.op=="remote") {assert(len);completed[r].insert(i.event);if(path) ++activeRelay;}
                        auto src=i.src.eval(e),target=i.dst.eval(e);
                        assert(src>=reinterpret_cast<uint64_t>(input[r].data()+guard)&&src+len<=reinterpret_cast<uint64_t>(input[r].data()+guard)+layouts[r].sendBytes);
                        assert(target>=reinterpret_cast<uint64_t>(output[dst].data()+guard)&&target+len<=reinterpret_cast<uint64_t>(output[dst].data()+guard)+layouts[dst].recvBytes);
                        if(len) std::memcpy(reinterpret_cast<void*>(target),reinterpret_cast<void*>(src),len);
                    }
                }
                ++pc[r];progress=true;
            }
            if(done) break;
            assert(progress && ++ticks<10000);
        }
        for(uint32_t d=0;d<8;++d) {
            size_t ro=0;
            for(uint32_t s=0;s<8;++s) {
                size_t so=0;for(uint32_t q=0;q<d;++q) so+=c[s*8+q]*rb;
                auto bytes=c[s*8+d]*rb;assert(!std::memcmp(output[d].data()+guard+ro,input[s].data()+guard+so,bytes));ro+=bytes;
            }
            for(size_t b=0;b<output[d].size();++b) if(b<guard||b>=guard+layouts[d].recvBytes) assert(output[d][b]==0xc3);
            assert(input[d]==original[d] && conditions[d].empty());
            for(auto& peer:bits[d]) for(auto& path:peer) for(auto v:path) assert(v==0);
        }
    }
    assert(activeRelay && skipped);
    std::cout<<"EP8_EID_GRAPH_PASS epochs=100 launches=800 channels_per_rank=49 task_args=46 relay_writes="<<activeRelay<<" skipped="<<skipped<<"\n";
}
