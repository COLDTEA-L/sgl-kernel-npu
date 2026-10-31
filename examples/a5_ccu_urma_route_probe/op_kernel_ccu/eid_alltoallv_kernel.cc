#include "eid_alltoallv_kernel.h"
#include "../common/path_layout.h"
#include <cstdio>
namespace a5_ccu_urma_probe {
#define E_CHECK(call) do { auto status = (call); if (status != HCCL_SUCCESS) return status; } while (0)
a5ccu::CcuKernelSignature EidAllToAllVKernelArg::GetKernelSignature() const {
    a5ccu::CcuKernelSignature s;
    std::vector<uint32_t> ids;
    for (const auto& p : paths) ids.push_back(p.peer*9+static_cast<uint32_t>(p.relay+1));
    s.Append(MakePathKernelSignature("A5CcuEidAllToAllV8V4DirectControl", ids, channels)); s.Append(EidRegisterStageName(stage)); s.Append(ready ? "ready_v1" : "immediate"); return s;
}
HcclResult EidAllToAllVKernel::Algorithm() {
    if (channels_.size() != paths_.size() || !ValidateEidChannels(rank_, paths_)) return HCCL_E_PARA;
    std::fprintf(stderr, "EP8_EID_GRAPH_BUILD rank=%u stage=%s channels=%zu registration_only=%u graph=%s\n",
        rank_, EidRegisterStageName(stage_), channels_.size(), !EidStageAllowsPayload(stage_),
        (stage_ == EidRegisterStage::MASK || stage_ >= EidRegisterStage::MASK_INIT) ? "legacy_mask_probe" : "direct_peer_control");
    std::fprintf(stderr,"EP8_EID_SCHEDULE rank=%u schedule=%s max_relay_flows=%u\n",rank_,ready_?"ready":"immediate",ready_?1:56);
    if (stage_ == EidRegisterStage::MINIMAL) {
        auto value = CreateVariable(); value = uint64_t{1};
        return HCCL_SUCCESS;
    }
    auto input=CreateVariable(), output=CreateVariable(), inputToken=CreateVariable(), outputToken=CreateVariable();
    auto selfSource=CreateVariable(), selfDestination=CreateVariable(), active=CreateVariable();
    Load(input); Load(output); Load(inputToken); Load(outputToken); Load(selfSource); Load(selfDestination);
    std::array<a5ccu::CcuRep::Variable,4> go;
    for (auto& v:go) Load(v);
    Load(active);
    // 11 + 7*5 = 46 task arguments, independent of up to 49 cached channels.
    std::array<std::array<a5ccu::CcuRep::Variable,5>,7> peer;
    // Public Variable subtraction exists in headers but is unsupported by
    // the V1 instruction generator (repType 11). GeneArgs supplies direct bytes.
    for (size_t j=0;j<7;++j) {for(auto& v:peer[j]) Load(v);}
    std::fprintf(stderr, "EP8_EID_TASK_LAYOUT rank=%u task_args=46 slots=0:input,1:output,2:input_token,3:output_token,4:self_src,5:self_dst,6-9:self_copy,10:helper_rank_or_reserved_mask,11-45:seven_peers_x_send_offset_recv_offset_direct_bytes_relay_bytes_relay_rank slot10=ready_helper_or_probe_mask_or_reserved\n", rank_);
    if (stage_ == EidRegisterStage::ARGS) return HCCL_SUCCESS;
    const size_t n=channels_.size();
    // Legacy mask operations are registration-only probes. Production never
    // constructs this decoder: CANN 9.3 End rejects the old MASK graph.
    if (stage_ == EidRegisterStage::MASK || stage_ >= EidRegisterStage::MASK_INIT) {
        std::vector<a5ccu::CcuRep::Variable> live(n);
        std::array<a5ccu::CcuRep::Variable,7> probeOutput, probeToken;
        for(size_t j=0;j<7;++j) {probeOutput[j]=uint64_t{0};probeToken[j]=uint64_t{0};}
        auto negativeBit=CreateVariable();
        for(size_t end=n;end>0;--end) {
            const size_t i=end-1;
            const uint64_t bit=uint64_t{1}<<i;
            live[i]=uint64_t{0};
            if (stage_ == EidRegisterStage::MASK_STATIC) live[i]=uint64_t{1};
            if (stage_ == EidRegisterStage::MASK_NEGATIVE || stage_ == EidRegisterStage::MASK_ADD || stage_ == EidRegisterStage::MASK)
                negativeBit=uint64_t{0}-bit;
            if (stage_ == EidRegisterStage::MASK_COMPARE) {CCU_IF(active >= bit) {live[i]=uint64_t{1};}}
            // EQ/NE use the same bit constants and graph size as MASK_COMPARE.
            // ZERO changes only the GE immediate; SINGLE isolates one GE block.
            if (stage_ == EidRegisterStage::MASK_EQUAL) {CCU_IF(active == bit) {live[i]=uint64_t{1};}}
            if (stage_ == EidRegisterStage::MASK_NOT_EQUAL) {CCU_IF(active != bit) {live[i]=uint64_t{1};}}
            if (stage_ == EidRegisterStage::MASK_COMPARE_ZERO) {CCU_IF(active >= uint64_t{0}) {live[i]=uint64_t{1};}}
            if (stage_ == EidRegisterStage::MASK_COMPARE_SINGLE && i==0) {CCU_IF(active >= uint64_t{1}) {live[i]=uint64_t{1};}}
            if (stage_ == EidRegisterStage::MASK_ADD) active+=negativeBit;
            if (stage_ == EidRegisterStage::MASK) {CCU_IF(active >= bit) {live[i]=uint64_t{1};active+=negativeBit;}}
        }
        return HCCL_SUCCESS;
    }
    // Fixed, reciprocal control plane: one direct channel per peer. Metadata
    // and completion are independent of the selected data path, including
    // receive-only/zero-row peers. Slot 10 carries the helper rank in READY mode, otherwise the reserved mask.
    std::array<a5ccu::CcuRep::Variable,7> remoteOutput, remoteToken;
    std::vector<size_t> control;
    std::vector<a5ccu::CcuRep::Variable> channelOutputs, channelTokens;
    control.reserve(7);channelOutputs.reserve(7);channelTokens.reserve(7);
    for (size_t i=0;i<n;++i) if(paths_[i].relay<0) {
        control.push_back(i);
        channelOutputs.emplace_back(BorrowChannelVariable(channels_[i],0));
        channelTokens.emplace_back(BorrowChannelVariable(channels_[i],1));
    }
    if (stage_ == EidRegisterStage::CHANNEL_VARS) return HCCL_SUCCESS;
    for(auto i:control) {
        E_CHECK(NotifyRecord(channels_[i],0,0,output,1));
        E_CHECK(NotifyRecord(channels_[i],1,1,outputToken,1));
    }
    for(size_t q=0;q<control.size();++q) {
        auto i=control[q];
        auto j=paths_[i].peer<rank_?paths_[i].peer:paths_[i].peer-1;
        E_CHECK(NotifyWait(channels_[i],0,1));E_CHECK(NotifyWait(channels_[i],1,1));
        // const lvalue assignments emit runtime copies, not handle rebinding.
        const auto& channelOutput=channelOutputs[q];
        const auto& channelToken=channelTokens[q];
        remoteOutput[j]=channelOutput;remoteToken[j]=channelToken;
    }
    if (stage_ == EidRegisterStage::SYNC) return HCCL_SUCCESS;
    // At most two writes per peer (direct + one selected relay), regardless of
    // the number of cached channels. reserve() does NOT construct SDK Events.
    std::vector<a5ccu::CcuRep::CompletedEvent> events;
    events.reserve(14);
    for(size_t i=0;i<14;++i) events.emplace_back(CreateCompletedEvent());
    std::array<a5ccu::CcuRep::LocalAddr,14> sources;
    std::array<a5ccu::CcuRep::RemoteAddr,14> destinations;
    for(size_t j=0;j<7;++j) for(size_t kind=0;kind<2;++kind) {
        auto& src=sources[j*2+kind];auto& dst=destinations[j*2+kind];
        src.addr=input;src.addr+=peer[j][0];src.token=inputToken;
        dst.addr=remoteOutput[j];dst.addr+=peer[j][1];dst.token=remoteToken[j];
        if(kind) {src.addr+=peer[j][2];dst.addr+=peer[j][2];}
    }
    // Submit every direct write before waiting. A helper must not depend on
    // the completion of an unrelated hot direct write at its source.
    for(size_t i=0;i<n;++i) if(paths_[i].relay<0) {
        auto j=paths_[i].peer<rank_?paths_[i].peer:paths_[i].peer-1;
        CCU_IF(peer[j][2] != 0) {E_CHECK(WriteNb(channels_[i],destinations[j*2],sources[j*2],peer[j][2],events[j*2]));}
    }
    if(ready_ && stage_ != EidRegisterStage::DIRECT) {
        // active is the host-selected helper rank, or 8 for no relay. EQ/NE
        // only: no mask decoder, GE, device subtraction, timers or host polling.
        for(uint32_t k=0;k<8;++k) {CCU_IF(active == uint64_t(k)) {
            if(rank_==k) {
                for(size_t j=0;j<7;++j) {CCU_IF(peer[j][2] != 0) {E_CHECK(WaitEvent(events[j*2]));}}
                for(auto i:control) {E_CHECK(NotifyWait(channels_[i],2,2));}
                for(auto i:control) {E_CHECK(NotifyRecord(channels_[i],2,4));}
            } else {
                const size_t j=k<rank_?k:k-1;
                // Wait only for this source's data destined for the helper.
                CCU_IF(peer[j][2] != 0) {E_CHECK(WaitEvent(events[j*2]));}
                for(auto i:control) if(paths_[i].peer==k) {
                    E_CHECK(NotifyRecord(channels_[i],2,2));
                    E_CHECK(NotifyWait(channels_[i],2,4));
                }
            }
        }}
    }
    if(stage_ != EidRegisterStage::DIRECT) for(size_t i=0;i<n;++i) if(paths_[i].relay>=0) {
        auto j=paths_[i].peer<rank_?paths_[i].peer:paths_[i].peer-1;
        CCU_IF(peer[j][4] == static_cast<uint64_t>(paths_[i].relay)) {
            CCU_IF(peer[j][3] != 0) {E_CHECK(WriteNb(channels_[i],destinations[j*2+1],sources[j*2+1],peer[j][3],events[j*2+1]));}
        }
    }
    if (stage_ == EidRegisterStage::DIRECT || stage_ == EidRegisterStage::RELAY) return HCCL_SUCCESS;
    auto src=CreateLocalAddr(), dst=CreateLocalAddr();
    src.addr=input;src.addr+=selfSource;src.token=inputToken;
    dst.addr=output;dst.addr+=selfDestination;dst.token=outputToken;
    E_CHECK(GroupedLocalCopy(dst,src,go));
    if (stage_ == EidRegisterStage::SELF) return HCCL_SUCCESS;
    for(size_t j=0;j<7;++j) {
        if(ready_) {
            const uint64_t p=j<rank_?j:j+1;
            // Events waited during READY are consumed; never wait twice.
            CCU_IF(active != uint64_t(rank_)) {CCU_IF(active != p) {
                CCU_IF(peer[j][2] != 0) {E_CHECK(WaitEvent(events[j*2]));}
            }}
        } else {CCU_IF(peer[j][2] != 0) {E_CHECK(WaitEvent(events[j*2]));}}
        CCU_IF(peer[j][3] != 0) {E_CHECK(WaitEvent(events[j*2+1]));}
    }
    for(auto i:control) {E_CHECK(NotifyRecord(channels_[i],2,1));}
    for(auto i:control) {E_CHECK(NotifyWait(channels_[i],2,1));}
    return HCCL_SUCCESS;
}
std::vector<uint64_t> EidAllToAllVKernel::GeneArgs(const a5ccu::CcuTaskArg& arg) {
    if (!EidStageAllowsPayload(stage_)) return {};
    const auto* t=dynamic_cast<const EidAllToAllVTaskArg*>(&arg);
    if(!t || t->layout.pathBytes.size()!=channels_.size() || (t->layout.activeMask>>channels_.size())) return {};
    if(ready_ && t->layout.readyRelay>8) return {};
    std::array<uint64_t,4> go{};
    if(t->layout.selfBytes && !BuildGroupCopyPlan(t->layout.selfBytes,&go)) return {};
    std::vector<uint64_t> args{t->input,t->output,t->inputToken,t->outputToken,
                             t->layout.selfSourceOffset,t->layout.selfDestinationOffset};
    args.insert(args.end(),go.begin(),go.end());args.push_back(ready_ ? t->layout.readyRelay : t->layout.activeMask);
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
