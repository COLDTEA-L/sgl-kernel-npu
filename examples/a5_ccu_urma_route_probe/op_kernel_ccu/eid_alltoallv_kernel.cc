#include "eid_alltoallv_kernel.h"
#include "../common/path_layout.h"
namespace a5_ccu_urma_probe {
#define E_CHECK(call) do { auto status = (call); if (status != HCCL_SUCCESS) return status; } while (0)
a5ccu::CcuKernelSignature EidAllToAllVKernelArg::GetKernelSignature() const {
    a5ccu::CcuKernelSignature s;
    std::vector<uint32_t> ids;
    for (const auto& p : paths) ids.push_back(p.peer*9+static_cast<uint32_t>(p.relay+1));
    s.Append(MakePathKernelSignature("A5CcuEidAllToAllV8V2AddCompare", ids, channels)); return s;
}
HcclResult EidAllToAllVKernel::Algorithm() {
    if (channels_.size() != paths_.size() || !ValidateEidChannels(rank_, paths_)) return HCCL_E_PARA;
    auto input=CreateVariable(), output=CreateVariable(), inputToken=CreateVariable(), outputToken=CreateVariable();
    auto selfSource=CreateVariable(), selfDestination=CreateVariable(), active=CreateVariable();
    Load(input); Load(output); Load(inputToken); Load(outputToken); Load(selfSource); Load(selfDestination);
    std::array<a5ccu::CcuRep::Variable,4> go;
    for (auto& v:go) Load(v);
    Load(active);
    // 11 + 7*5 = 46 task arguments, independent of up to 49 cached channels.
    std::array<std::array<a5ccu::CcuRep::Variable,5>,7> peer;
    std::array<a5ccu::CcuRep::Variable,7> direct;
    // Public Variable subtraction exists in headers but is unsupported by
    // the V1 instruction generator (repType 11). GeneArgs supplies direct bytes.
    for (size_t j=0;j<7;++j) {for(auto& v:peer[j]) Load(v); direct[j]=peer[j][2];}
    const size_t n=channels_.size();
    std::vector<a5ccu::CcuRep::Variable> live(n), remoteOutput(n), remoteToken(n), bytes(n);
    std::vector<a5ccu::CcuRep::CompletedEvent> events(n);
    // V1 also lacks variable AND. Decode the <=49-bit mask from high to low
    // using unsigned comparison and modulo-2^64 addition, both V1 primitives.
    // A host-computed negative constant is not a CCU subtraction instruction.
    auto negativeBit=CreateVariable();
    for(size_t end=n;end>0;--end) {
        const size_t i=end-1;
        const uint64_t bit=uint64_t{1}<<i;
        live[i]=uint64_t{0};
        negativeBit=uint64_t{0}-bit;
        CCU_IF(active >= bit) { live[i]=uint64_t{1}; active+=negativeBit; }
    }
    for(size_t i=0;i<n;++i) {
        auto j=paths_[i].peer<rank_?paths_[i].peer:paths_[i].peer-1;
        if(paths_[i].relay<0) bytes[i]=direct[j];
        else bytes[i]=peer[j][3];
        E_CHECK(CreateVariable(channels_[i],0,&remoteOutput[i]));
        E_CHECK(CreateVariable(channels_[i],1,&remoteToken[i]));
        events[i]=CreateCompletedEvent();
        // Both directions use the union of active paths, including receive-only ranks.
        CCU_IF(live[i] != 0) {
            E_CHECK(NotifyRecord(channels_[i],0,0,output,1));
            E_CHECK(NotifyRecord(channels_[i],1,1,outputToken,1));
        }
    }
    for(size_t i=0;i<n;++i) {CCU_IF(live[i] != 0) {
        E_CHECK(NotifyWait(channels_[i],0,1)); E_CHECK(NotifyWait(channels_[i],1,1));
    }}
    for(size_t i=0;i<n;++i) {
        auto j=paths_[i].peer<rank_?paths_[i].peer:paths_[i].peer-1;
        auto src=CreateLocalAddr(); auto dst=CreateRemoteAddr();
        src.addr=input; src.addr+=peer[j][0]; src.token=inputToken;
        dst.addr=remoteOutput[i]; dst.addr+=peer[j][1]; dst.token=remoteToken[i];
        if(paths_[i].relay>=0) {src.addr+=direct[j];dst.addr+=direct[j];}
        if(paths_[i].relay<0) {
            CCU_IF(bytes[i] != 0) {E_CHECK(WriteNb(channels_[i],dst,src,bytes[i],events[i]));}
        } else {CCU_IF(peer[j][4] == static_cast<uint64_t>(paths_[i].relay)) {
            CCU_IF(bytes[i] != 0) {E_CHECK(WriteNb(channels_[i],dst,src,bytes[i],events[i]));}
        }}
    }
    auto src=CreateLocalAddr(), dst=CreateLocalAddr();
    src.addr=input;src.addr+=selfSource;src.token=inputToken;
    dst.addr=output;dst.addr+=selfDestination;dst.token=outputToken;
    E_CHECK(GroupedLocalCopy(dst,src,go));
    for(size_t i=0;i<n;++i) {
        auto j=paths_[i].peer<rank_?paths_[i].peer:paths_[i].peer-1;
        if(paths_[i].relay<0) {CCU_IF(bytes[i] != 0) {E_CHECK(WaitEvent(events[i]));}}
        else {CCU_IF(peer[j][4] == static_cast<uint64_t>(paths_[i].relay)) {
            CCU_IF(bytes[i] != 0) {E_CHECK(WaitEvent(events[i]));}
        }}
    }
    for(size_t i=0;i<n;++i) {CCU_IF(live[i] != 0) {E_CHECK(NotifyRecord(channels_[i],2,1));}}
    for(size_t i=0;i<n;++i) {CCU_IF(live[i] != 0) {E_CHECK(NotifyWait(channels_[i],2,1));}}
    return HCCL_SUCCESS;
}
std::vector<uint64_t> EidAllToAllVKernel::GeneArgs(const a5ccu::CcuTaskArg& arg) {
    const auto* t=dynamic_cast<const EidAllToAllVTaskArg*>(&arg);
    if(!t || t->layout.pathBytes.size()!=channels_.size() || (t->layout.activeMask>>channels_.size())) return {};
    std::array<uint64_t,4> go{};
    if(t->layout.selfBytes && !BuildGroupCopyPlan(t->layout.selfBytes,&go)) return {};
    std::vector<uint64_t> args{t->input,t->output,t->inputToken,t->outputToken,
                             t->layout.selfSourceOffset,t->layout.selfDestinationOffset};
    args.insert(args.end(),go.begin(),go.end());args.push_back(t->layout.activeMask);
    for(const auto& p:t->layout.peerArgs) {
        if(p[3]>p[2]) return {};
        args.insert(args.end(),{p[0],p[1],p[2]-p[3],p[3],p[4]});
    }
    return args;
}
std::unique_ptr<a5ccu::CcuKernel> CreateEidAllToAllVKernel(const a5ccu::CcuKernelArg& a) {
    return std::unique_ptr<a5ccu::CcuKernel>(new EidAllToAllVKernel(a));
}
#undef E_CHECK
}
