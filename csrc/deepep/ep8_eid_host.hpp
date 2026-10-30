#pragma once
#include "ccu_payload.hpp"
#include "ep8_relay_planner.hpp"
#include <ATen/ATen.h>
#include <chrono>
#include <memory>
#include <mutex>

namespace deep_ep::ep8 {
struct EidRuntime {
    virtual ~EidRuntime() = default;
    virtual void Check(const at::Tensor&) = 0;
    virtual void Sync() = 0;
    virtual void Transfer(const at::Tensor&, const at::Tensor&, const Plan&, U rowBytes) = 0;
};
struct EidPrepared {
    Plan plan;
    U rowBytes=0, sendRows=0, recvRows=0;
    const void* owner=nullptr;
    at::ScalarType dtype=at::kByte;
    c10::Device device=c10::Device(c10::kCPU);
    std::vector<int64_t> shape;
    bool UsesRelay() const { return !plan.actions.empty(); }
};
// One original caller stream and one registered end-to-end CCU kernel. No
// participant HBM relay, staging tensors, per-chunk launch or additional group.
class EidHost {
public:
    EidHost(const Planner& planner, int rank, std::shared_ptr<EidRuntime> runtime, U maxPayloads=4)
        : planner_(planner), rank_(rank), rt_(std::move(runtime)), maxPayloads_(maxPayloads) {
        TORCH_CHECK(rank>=0 && rank<8 && rt_ && maxPayloads, "invalid EP8 EID host configuration");
    }
    ~EidHost() {try {Close();} catch(const std::exception& e) {std::fprintf(stderr,"EP8_EID_CLOSE_FAILED %s\n",e.what());}}
    std::shared_ptr<EidPrepared> Prepare(const Matrix& c, const at::Tensor& example) {
        auto lock=Lock();Check();auto& payload=Select(example);
        auto p=std::make_shared<EidPrepared>();Build(c,example,payload,*p);return p;
    }
    void Execute(const std::shared_ptr<EidPrepared>& p,const at::Tensor& send,const at::Tensor& recv) {
        auto lock=Lock();Check();
        TORCH_CHECK(p && p->owner==this,"EID prepared belongs to another host");
        Validate(*p,send,recv); auto& payload=Select(send);Submit(*p,payload,send,recv);
    }
    std::array<double,3> Run(const Matrix& c,const at::Tensor& send,const at::Tensor& recv) {
        auto lock=Lock();Check();auto start=Clock::now();auto& payload=Select(send);
        Build(c,send,payload,payload.dynamic);Validate(payload.dynamic,send,recv);
        auto planned=Micros(start);start=Clock::now();Submit(payload.dynamic,payload,send,recv);
        return {planned,0.,Micros(start)};
    }
    void Close() {
        auto lock=Lock();if(closed_) return;rt_->Sync();payloads_.clear();closed_=true;
    }
    // Same seven counters as pipeline: no workspace or event allocation here.
    std::array<U,7> Stats() const {
        auto lock=Lock();U bytes=0;
        for(const auto& p:payloads_) bytes+=p.zeroSend.nbytes()+p.zeroRecv.nbytes();
        return {0,allocations_,0,0,poisoned_,bytes,closed_};
    }
    struct PayloadInfo {std::string device,dtype;std::vector<int64_t> shape;U rowBytes;};
    std::vector<PayloadInfo> PayloadStats() const {
        auto lock=Lock();std::vector<PayloadInfo> out;
        for(const auto& p:payloads_) out.push_back({p.zeroSend.device().str(),c10::toString(p.zeroSend.scalar_type()),p.shape,p.planner.RowBytes()});
        return out;
    }
private:
    struct Payload {Planner planner;std::vector<int64_t> shape;at::Tensor zeroSend,zeroRecv;EidPrepared dynamic;};
    using Clock=std::chrono::steady_clock;
    static double Micros(Clock::time_point start) {return std::chrono::duration<double,std::micro>(Clock::now()-start).count();}
    std::unique_lock<std::mutex> Lock() const {
        std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
        TORCH_CHECK(lock.owns_lock(),"EID host does not allow concurrent submissions");return lock;
    }
    void Check() const {TORCH_CHECK(!closed_ && !poisoned_,"EID host closed or failed; recreate context");}
    Payload& Select(const at::Tensor& t) {
        rt_->Check(t);
        TORCH_CHECK(t.dim()>=1 && t.is_contiguous() && IsCcuPayload(t.scalar_type()),"invalid EID payload");
        for(auto& p:payloads_) if(t.device()==p.zeroSend.device() && t.scalar_type()==p.zeroSend.scalar_type() &&
                                  t.sizes().slice(1)==at::IntArrayRef(p.shape)) return p;
        U bytes=t.element_size();
        for(int i=1;i<t.dim();++i) {TORCH_CHECK(t.size(i)>0 && static_cast<U>(t.size(i))<=Max/bytes,"invalid row shape");bytes*=t.size(i);}
        TORCH_CHECK(payloads_.size()<maxPayloads_,"EID max_payloads exceeded");
        auto shape=t.sizes().vec();shape[0]=1;
        payloads_.push_back({planner_.WithRowBytes(bytes),t.sizes().slice(1).vec(),
                            at::empty(shape,t.options()),at::empty(shape,t.options()),{}});
        allocations_+=2;return payloads_.back();
    }
    void Build(const Matrix& c,const at::Tensor& t,Payload& payload,EidPrepared& out) {
        payload.planner.MakeInto(c,out.plan);
        out.sendRows=out.recvRows=0;
        for(int k=0;k<8;++k) {out.sendRows+=c[rank_*8+k];out.recvRows+=c[k*8+rank_];}
        TORCH_CHECK(static_cast<U>(t.size(0))==out.sendRows,"EID send rows differ from counts");
        out.rowBytes=payload.planner.RowBytes();out.dtype=t.scalar_type();out.device=t.device();
        out.shape=payload.shape;out.owner=this;
    }
    void Validate(const EidPrepared& p,const at::Tensor& send,const at::Tensor& recv) {
        rt_->Check(send);
        TORCH_CHECK(send.dim()>=1 && recv.dim()==send.dim() && send.is_contiguous() && recv.is_contiguous() &&
                    send.device()==p.device && recv.device()==p.device && send.scalar_type()==p.dtype && recv.scalar_type()==p.dtype &&
                    send.sizes().slice(1)==at::IntArrayRef(p.shape) && recv.sizes().slice(1)==at::IntArrayRef(p.shape) &&
                    static_cast<U>(send.size(0))==p.sendRows && static_cast<U>(recv.size(0))==p.recvRows,"EID prepared payload mismatch");
        if(send.numel() && recv.numel()) {
            auto a=reinterpret_cast<uintptr_t>(send.data_ptr()),b=reinterpret_cast<uintptr_t>(recv.data_ptr());
            TORCH_CHECK(a<=b? b-a>=send.nbytes(): a-b>=recv.nbytes(),"EID buffers overlap");
        }
    }
    void Submit(const EidPrepared& p,Payload& payload,const at::Tensor& send,const at::Tensor& recv) {
        auto src=send.storage().nbytes()?send:payload.zeroSend.narrow(0,0,0);
        auto dst=recv.storage().nbytes()?recv:payload.zeroRecv.narrow(0,0,0);
        try {rt_->Transfer(src,dst,p.plan,p.rowBytes);} catch(...) {poisoned_=true;throw;}
    }
    Planner planner_;int rank_;std::shared_ptr<EidRuntime> rt_;U maxPayloads_,allocations_=0;
    bool closed_=false,poisoned_=false;
    mutable std::mutex mutex_;std::vector<Payload> payloads_;
};
}
