#include "../../csrc/deepep/ep8_eid_host.hpp"
#include "../../examples/a5_ccu_urma_route_probe/common/eid_alltoallv_layout.h"
#include <cstring>
#include <iostream>
using namespace deep_ep;
using namespace deep_ep::ep8;
using namespace a5_ccu_urma_probe;
static void Require(bool ok) {if(!ok) throw std::runtime_error("EID host check failed");}
template<class F> static void Reject(F f) {bool caught=false;try{f();}catch(const std::exception&){caught=true;}Require(caught);}
struct Batch {
    struct Call {at::Tensor src,dst;Plan plan;U rb;};
    std::array<Call,8> calls;std::array<bool,8> ready{};std::array<uint8_t,64> allowed{};
    void Run() {
        for(bool r:ready) Require(r);
        for(int s=0;s<8;++s) {
            auto& call=calls[s];Require(call.plan.counts==calls[0].plan.counts && call.plan.relays==calls[0].plan.relays && call.rb==calls[0].rb);
            std::vector<EidChannel> channels;
            for(int d=0;d<8;++d) if(s!=d) {channels.push_back({static_cast<uint32_t>(d),-1});
                for(int k=0;k<8;++k) if(k!=s&&k!=d) channels.push_back({static_cast<uint32_t>(d),k});}
            EidAllToAllVLayout l;Require(BuildEidAllToAllVLayout(s,call.plan.counts.data(),64,call.rb,
                call.plan.relays.data(),call.plan.moved.data(),allowed,channels,&l));
            for(size_t i=0;i<channels.size();++i) if(l.pathBytes[i]) {
                auto d=channels[i].peer;
                std::memcpy(static_cast<char*>(calls[d].dst.data_ptr())+l.remoteOffsets[i],
                            static_cast<const char*>(call.src.data_ptr())+l.sourceOffsets[i],l.pathBytes[i]);
            }
            if(l.selfBytes) std::memcpy(static_cast<char*>(call.dst.data_ptr())+l.selfDestinationOffset,
                                        static_cast<const char*>(call.src.data_ptr())+l.selfSourceOffset,l.selfBytes);
        }
        ready.fill(false);for(auto& c:calls) c=Call{};
    }
};
struct Runtime:EidRuntime {
    Batch& batch;int rank;int submits=0,syncs=0;bool fail=false;
    Runtime(Batch& b,int r):batch(b),rank(r){}
    void Check(const at::Tensor& t) override {Require(t.device().is_cpu());}
    void Sync() override {++syncs;for(bool r:batch.ready) Require(!r);}
    void Transfer(const at::Tensor& src,const at::Tensor& dst,const Plan& p,U rb) override {
        if(fail) throw std::runtime_error("injected launch failure");
        Require(!batch.ready[rank]);++submits;batch.calls[rank]={src,dst,p,rb};batch.ready[rank]=true;
    }
};
struct Payload {
    std::array<at::Tensor, 8> sends, stores, outputs, expected;
};
static Payload Make(const Matrix& c, at::ScalarType type, int width, int epoch) {
    Payload p; U rb = width * c10::elementSize(type);
    for (int r = 0; r < 8; ++r) {
        U nr = 0, ns = 0;
        for (int k = 0; k < 8; ++k) { ns += c[r*8+k]; nr += c[k*8+r]; }
        p.sends[r] = at::empty({static_cast<int64_t>(ns), width}, at::TensorOptions().dtype(type));
        auto data = static_cast<uint8_t*>(p.sends[r].data_ptr());
        for (U j = 0; j < ns*rb; ++j) data[j] = (j*13+r*29+epoch*17) % 256;
        // Row guards keep typed views aligned, including BF16.
        p.stores[r] = at::full({static_cast<int64_t>((nr+2)*rb)}, 239, at::TensorOptions().dtype(at::kByte));
        p.outputs[r] = p.stores[r].narrow(0, rb, nr*rb).view(type).view({static_cast<int64_t>(nr), width});
        p.expected[r] = p.stores[r].clone();
    }
    for (int s = 0; s < 8; ++s) for (int d = 0; d < 8; ++d) {
        U so = 0, ro = 0, n = c[s*8+d];
        for (int k = 0; k < d; ++k) so += c[s*8+k];
        for (int k = 0; k < s; ++k) ro += c[k*8+d];
        if (n) std::memcpy(p.expected[d].data_ptr<uint8_t>()+(ro+1)*rb,
                           static_cast<const uint8_t*>(p.sends[s].data_ptr())+so*rb, n*rb);
    }
    return p;
}
static Matrix Transpose(const Matrix& c) { Matrix t{}; for (int s=0;s<8;++s) for(int d=0;d<8;++d) t[d*8+s]=c[s*8+d]; return t; }
static void Verify(const Payload& p) { for (int r=0;r<8;++r) Require(at::equal(p.stores[r],p.expected[r])); }

static void Run(bool dense) {
    Batch batch;std::vector<std::vector<U>> allowed(64);
    for(int s=0;s<8;++s)for(int d=0;d<8;++d)if(s!=d)for(int k=0;k<8;++k)if(k!=s&&k!=d){allowed[s*8+d].push_back(k);batch.allowed[s*8+d]|=1u<<k;}
    Planner planner(allowed,5120,1,0,56,{});
    if(dense) planner=planner.WithDenseReady(8);
    Require(planner.WithRowBytes(10240).DenseDivisor()==(dense?8:0));
    std::array<std::shared_ptr<Runtime>,8> runtimes;std::array<std::unique_ptr<EidHost>,8> hosts;
    Matrix c{};c[4]=65;c[0]=3;
    auto warm=Make(c,at::kFloat8_e4m3fn,5120,0),rev=Make(Transpose(c),at::kBFloat16,5120,0);
    std::array<std::shared_ptr<EidPrepared>,8> prepared;
    for(int r=0;r<8;++r){runtimes[r]=std::make_shared<Runtime>(batch,r);hosts[r]=std::make_unique<EidHost>(planner,r,runtimes[r]);
        prepared[r]=hosts[r]->Prepare(c,warm.sends[r]);hosts[r]->Prepare(Transpose(c),rev.sends[r]);Require(prepared[r]->UsesRelay());}
    for(int epoch=0;epoch<100;++epoch) {
        auto counts=c;if(epoch%10==0) counts={};else if(epoch%10==1) {counts={};for(int r=0;r<8;++r) counts[r*8+r]=r+1;}
        else if(epoch%10>=5) for(int s=0;s<8;++s)for(int d=0;d<8;++d) counts[s*8+d]=(s*7+d*3+epoch)%9;
        auto a=Make(counts,at::kFloat8_e4m3fn,5120,epoch),b=Make(Transpose(counts),at::kBFloat16,5120,epoch+1);
        for(int r=0;r<8;++r)hosts[r]->Run(counts,a.sends[r],a.outputs[r]);batch.Run();Verify(a);
        for(int r=0;r<8;++r)hosts[r]->Run(Transpose(counts),b.sends[r],b.outputs[r]);batch.Run();Verify(b);
    }
    for(int r=0;r<8;++r){Require(runtimes[r]->submits==200&&runtimes[r]->syncs==0);
        auto s=hosts[r]->Stats();Require(s[1]==4&&s[3]==0&&s[5]==2*(5120+10240));
        Reject([&]{hosts[r]->Execute(prepared[r],rev.sends[r],rev.outputs[r]);});
        hosts[r]->Execute(prepared[r],warm.sends[r],warm.outputs[r]);}
    batch.Run();Verify(warm);
    runtimes[0]->fail=true;Reject([&]{hosts[0]->Run(c,warm.sends[0],warm.outputs[0]);});Require(hosts[0]->Stats()[4]);
    Reject([&]{hosts[0]->Run(Transpose(c),rev.sends[0],rev.outputs[0]);});
    for(auto& h:hosts){h->Close();Require(h->Stats()[6]);}
    std::cout<<"EP8_EID_HOST_PASS ranks=8 alternating=100 row_bytes=5120,10240 submissions_per_call=1 relay_hbm_bytes=0 dense="<<dense<<"\n";
}

int main() { Run(false); Run(true); }
