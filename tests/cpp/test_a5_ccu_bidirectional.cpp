// Production C++ planner/layout/host with a deferred eight-rank CPU transport.
// Reorders ready work across lanes; checks bytes/guards, not floating arithmetic.
#include "../../csrc/deepep/ep8_pipeline_host.hpp"
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
using namespace deep_ep;
using namespace deep_ep::ep8;
static void Require(bool ok) { if (!ok) throw std::runtime_error("bidirectional check failed"); }
template<class F> static void Reject(F f) {
    bool caught = false; try { f(); } catch (const std::exception&) { caught = true; } Require(caught);
}
struct Call { at::Tensor send, recv; Matrix counts{}, storage{}; U bytes = 0; };
struct Node {
    std::vector<int> deps; std::function<void()> action;
    std::array<Call, 8> calls; int arrivals = 0; bool collective = false, done = false;
};
struct Engine {
    std::vector<std::shared_ptr<Node>> nodes;
    std::map<std::pair<int, int>, int> collectives;
    std::array<std::array<int, 3>, 8> tails;
    Engine() { for (auto& t : tails) t.fill(-1); }
    int Add(int rank, int lane, std::function<void()> action = {}) {
        auto n = std::make_shared<Node>(); n->deps.push_back(tails[rank][lane]); n->action = std::move(action);
        nodes.push_back(n); return tails[rank][lane] = nodes.size()-1;
    }
    void Run() {
        bool progress;
        do {
            progress = false;
            // Reverse traversal deliberately gives later relay work priority.
            for (int i = static_cast<int>(nodes.size())-1; i >= 0; --i) {
                auto& n = *nodes[i]; if (n.done || (n.collective && n.arrivals != 8)) continue;
                bool ready = true;
                for (int d : n.deps) if (d >= 0 && !nodes[d]->done) ready = false;
                if (!ready) continue;
                if (n.collective) {
                    for (int s = 0; s < 8; ++s) for (int d = 0; d < 8; ++d) {
                        auto& from = n.calls[s]; auto& to = n.calls[d];
                        Require(from.bytes == to.bytes && from.counts == to.counts && from.storage == to.storage);
                        U rows = from.counts[s*8+d], so = 0, ro = 0;
                        for (int k = 0; k < d; ++k) so += from.storage[s*8+k];
                        for (int k = 0; k < s; ++k) ro += to.storage[k*8+d];
                        if (rows) {
                            Require((so+rows)*from.bytes <= from.send.nbytes() && (ro+rows)*from.bytes <= to.recv.nbytes());
                            std::memcpy(static_cast<char*>(to.recv.data_ptr()) + ro*from.bytes,
                                        static_cast<const char*>(from.send.data_ptr()) + so*from.bytes, rows*from.bytes);
                        }
                    }
                } else if (n.action) n.action();
                n.done = true; n.action = {}; n.calls = {}; progress = true;
            }
        } while (progress);
        for (auto& n : nodes) Require(n->done); // no hidden cyclic dependency
    }
};
struct Runtime : PipelineRuntime {
    Engine& e; int rank, active = 0, syncs = 0, transfers = 0;
    std::array<int, 3> seq{}; std::vector<int> events; bool fail = false;
    Runtime(Engine& engine, int r) : e(engine), rank(r) {}
    void Check(const at::Tensor& t) override { Require(t.device().is_cpu()); }
    void Sync() override { ++syncs; e.Run(); }
    void Lane(int lane) override { active = lane; }
    void Events(size_t n) override { if (events.size() < n) events.resize(n, -1); }
    void Record(size_t id, int lane) override { events.at(id) = e.Add(rank, lane); }
    void Wait(size_t id, int lane) override { auto node = e.Add(rank, lane); e.nodes[node]->deps.push_back(events.at(id)); }
    void Retain(const at::Tensor&, int) override {}
    void Copy(const at::Tensor& src, const at::Tensor& dst, const CopySpans& spans) override {
        e.Add(rank, active, [src, dst, spans] { CopySpansOut(src, dst, spans); });
    }
    void Transfer(const at::Tensor& src, const at::Tensor& dst, const Matrix& c, const Matrix* storage, U bytes) override {
        if (fail) throw std::runtime_error("injected shared runtime failure");
        ++transfers; auto key = std::make_pair(active, seq[active]++);
        auto it = e.collectives.find(key);
        if (it == e.collectives.end()) {
            int id = e.nodes.size(); e.nodes.push_back(std::make_shared<Node>()); e.nodes.back()->collective = true;
            it = e.collectives.emplace(key, id).first;
        }
        auto& node = *e.nodes[it->second]; node.deps.push_back(e.tails[rank][active]); ++node.arrivals;
        node.calls[rank] = {src, dst, c, storage ? *storage : c, bytes}; e.tails[rank][active] = it->second;
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
int main() {
    std::vector<std::vector<U>> allowed(64);
    for (int s=0;s<8;++s) for(int d=0;d<8;++d) if(s!=d) for(U k=0;k<8;++k)
        if(k!=static_cast<U>(s)&&k!=static_cast<U>(d)) allowed[s*8+d].push_back(k);
    // Rebinding must retain byte-valued limits including non-row-aligned limits.
    Matrix c{}; c[4]=65; c[0]=3; c[9]=2;
    Planner seed(allowed,5120,5131,11,56,std::vector<U>(8,120003));
    for (U rb : {10240u, 4u, 5120u}) {
        auto a=seed.WithRowBytes(rb).Make(c), b=Planner(allowed,rb,5131,11,56,std::vector<U>(8,120003)).Make(c);
        Require(a.relays==b.relays && a.moved==b.moved);
    }
    for (bool bypass : {false, true}) {
        Engine engine; std::array<std::shared_ptr<Runtime>,8> runtimes;
        std::array<std::unique_ptr<PipelineHost>,8> hosts;
        Planner planner(allowed,5120,1,0,56,{});
        std::array<std::array<std::shared_ptr<PipelinePrepared>,8>,2> prepared;
        auto warm=Make(c,at::kFloat8_e4m3fn,5120,0), reverse=Make(Transpose(c),at::kBFloat16,5120,1);
        for(int r=0;r<8;++r) {
            runtimes[r]=std::make_shared<Runtime>(engine,r);
            hosts[r]=std::make_unique<PipelineHost>(planner,r,8*5120,128,1<<24,runtimes[r],bypass,3);
            prepared[0][r]=hosts[r]->Prepare(c,warm.sends[r]);
            prepared[1][r]=hosts[r]->Prepare(Transpose(c),reverse.sends[r]);
            Require(prepared[0][r]->rowBytes==5120 && prepared[1][r]->rowBytes==10240);
            Require(prepared[0][r]->UsesRelay() && prepared[1][r]->UsesRelay());
        }
        std::array<U,8> allocations; std::array<int,8> syncs;
        for(int r=0;r<8;++r) { allocations[r]=hosts[r]->Stats()[1];syncs[r]=runtimes[r]->syncs; }
        for(int epoch=0;epoch<100;++epoch) {
            Matrix m=c; m[4]-=epoch%3;
            auto a=Make(m,at::kFloat8_e4m3fn,5120,epoch), b=Make(Transpose(m),at::kBFloat16,5120,epoch+1);
            for(int r=0;r<8;++r) hosts[r]->Run(m,a.sends[r],a.outputs[r]);
            // Queue reverse direction before draining dispatch: shared events must still be safe.
            for(int r=0;r<8;++r) hosts[r]->Run(Transpose(m),b.sends[r],b.outputs[r]);
            engine.Run(); Verify(a);Verify(b);
        }
        for(int r=0;r<8;++r) {
            Require(hosts[r]->PayloadStats().size()==2 && hosts[r]->Stats()[1]==allocations[r] && runtimes[r]->syncs==syncs[r]);
            Reject([&]{hosts[r]->Execute(prepared[0][r],reverse.sends[r],reverse.outputs[r]);});
            hosts[r]->Execute(prepared[0][r],warm.sends[r],warm.outputs[r]);
        }
        for(int r=0;r<8;++r) hosts[r]->Execute(prepared[1][r],reverse.sends[r],reverse.outputs[r]);
        engine.Run();Verify(warm);Verify(reverse);
        // Empty, self-only, and asymmetric matrices in both directions.
        for(int kind=0;kind<3;++kind) {
            Matrix m{};
            for(int s=0;s<8;++s) for(int d=0;d<8;++d)
                m[s*8+d]=kind==1?(s==d?s+1:0):kind==2?(s*7+d*11)%5:0;
            auto a=Make(m,at::kFloat8_e4m3fn,5120,kind), b=Make(Transpose(m),at::kBFloat16,5120,kind+1);
            // Prepare all ranks before enqueueing if a new matrix grows a pool.
            for(int r=0;r<8;++r) {prepared[0][r]=hosts[r]->Prepare(m,a.sends[r]);prepared[1][r]=hosts[r]->Prepare(Transpose(m),b.sends[r]);}
            for(int r=0;r<8;++r) hosts[r]->Execute(prepared[0][r],a.sends[r],a.outputs[r]);
            for(int r=0;r<8;++r) hosts[r]->Execute(prepared[1][r],b.sends[r],b.outputs[r]);
            engine.Run();Verify(a);Verify(b);
        }
        // Equal bytes/row must not conflate different dtype/tail shapes.
        auto equalBytes=at::empty({0,2560},at::TensorOptions().dtype(at::kBFloat16));
        hosts[0]->Prepare(Matrix{},equalBytes);Require(hosts[0]->PayloadStats().size()==3);
        Reject([&]{hosts[0]->Prepare(Matrix{},at::empty({0,1280},at::TensorOptions().dtype(at::kFloat)));});
        // One failed lane poisons all typed payload profiles sharing the runtime.
        runtimes[0]->fail=true;
        Reject([&]{hosts[0]->Run(Matrix{},warm.sends[7],warm.outputs[7]);});
        Require(hosts[0]->Stats()[4]);
        Reject([&]{hosts[0]->Run(Matrix{},reverse.sends[7],reverse.outputs[7]);});
        for(auto& host:hosts) {host->Close();Require(host->Stats()[3]==0 && host->Stats()[6]);}
    }
    // Workspace budget is shared across signatures, rather than multiplied by their count.
    Engine e; auto rt=std::make_shared<Runtime>(e,0); Planner planner(allowed,5120,1,0,56,{});
    PipelineHost h(planner,0,8*5120,128,300000,rt,false);
    auto a=Make(c,at::kFloat8_e4m3fn,5120,0), b=Make(Transpose(c),at::kBFloat16,5120,0);
    h.Prepare(c,a.sends[0]);Reject([&]{h.Prepare(Transpose(c),b.sends[0]);});
    Require(h.Stats()[3]<=300000 && rt->transfers==0);
    std::cout<<"BIDIRECTIONAL_CPP_PASS ranks=8 iterations=100 dtypes=fp8,bf16 width=5120 bypass=off,on guards=PASS steady_allocations=0 steady_syncs=0\n";
}
