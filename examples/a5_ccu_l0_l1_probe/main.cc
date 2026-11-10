// Independent two-rank public-API probe. No EP8 code or route .so is linked.
#include "layout.h"
#include <acl/acl.h>
#include <hccl/hccl.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_rank_graph.h>
#include <hcomm/ccu/ccu_launch.h>
#include <hcomm/ccu/ccu_primitives.hpp>
#include <hcomm/ccu/ccu_res.h>
#include <hcomm/hcomm_res.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

namespace ccu = AscendC::ccu;
namespace {
using Clock = std::chrono::steady_clock;
void Check(long status, const char* expr) {
    if (status) throw std::runtime_error(std::string(expr) + " status=" + std::to_string(status));
}
#define CHECK(x) Check(static_cast<long>(x), #x)
#define EMIT(x) do { const auto s = (x); if (s != CcuResult::CCU_SUCCESS) return s; } while (0)
struct Options {
    unsigned rank=0, device=0, warmup=20, iters=50, rounds=6, l0Percent=50;
    unsigned l0Index=0, l1Index=0;
    uint64_t bytes=8ULL<<20;
    std::string dir, mode="all", suite="all", timing="block";
};
uint64_t Number(const std::string& s) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("invalid unsigned integer: " + s);
    return std::stoull(s);
}
Options Parse(int argc, char** argv) {
    Options o;
    for (int i=1; i<argc; i+=2) {
        if (i+1==argc) throw std::runtime_error("missing option value");
        std::string k=argv[i], v=argv[i+1];
        if(k=="--dir") o.dir=v;
        else if(k=="--mode") o.mode=v;
        else if(k=="--suite") o.suite=v;
        else if(k=="--timing") o.timing=v;
        else if(k=="--bytes") o.bytes=Number(v);
        else {
            auto n=Number(v);
            if(n>1000000) throw std::runtime_error("numeric option too large");
            if(k=="--rank") o.rank=n;
            else if(k=="--device") o.device=n;
            else if(k=="--warmup") o.warmup=n;
            else if(k=="--iters") o.iters=n;
            else if(k=="--rounds") o.rounds=n;
            else if(k=="--l0-percent") o.l0Percent=n;
            else if(k=="--l0-index") o.l0Index=n;
            else if(k=="--l1-index") o.l1Index=n;
            else throw std::runtime_error("unknown option " + k);
        }
    }
    if(o.rank>1 || o.device>1 || o.dir.empty() || !o.iters || !o.rounds || o.rounds>1000 ||
       (o.mode!="all" && o.mode!="l0" && o.mode!="l1" && o.mode!="split") ||
       (o.suite!="discover" && o.suite!="create" && o.suite!="correctness" && o.suite!="benchmark" && o.suite!="all") ||
       (o.timing!="block" && o.timing!="serial")) throw std::runtime_error("invalid options");
    l0l1::Split(o.bytes,o.l0Percent);
    return o;
}
void Phase(const Options& o, const std::string& stage) {
    std::cout << "L0L1_PHASE rank=" << o.rank << " stage=" << stage << std::endl;
}
void Save(const std::string& path, const std::string& data) {
    const auto tmp=path+".part";
    std::ofstream f(tmp,std::ios::binary); f.write(data.data(),data.size()); f.close();
    if(!f || std::rename(tmp.c_str(),path.c_str())) throw std::runtime_error("cannot publish " + path);
}
std::string ReadWait(const std::string& path) {
    auto deadline=Clock::now()+std::chrono::seconds(120);
    do {
        std::ifstream f(path,std::ios::binary);
        if(f) return std::string(std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>());
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } while(Clock::now()<deadline);
    throw std::runtime_error("peer file timeout " + path);
}
// All barriers are out of band and outside measured intervals, never HCCL collectives.
void Agree(const Options& o, const std::string& name, const std::string& value="ok") {
    const auto prefix=o.dir+"/"+name+".";
    Save(prefix+std::to_string(o.rank), value);
    if(ReadWait(prefix+std::to_string(1-o.rank))!=value)
        throw std::runtime_error("rank disagreement: " + name);
}
std::string EndpointKey(const EndpointDesc& ep) {
    std::ostringstream s;
    s << int(ep.protocol) << ":" << int(ep.commAddr.type) << ":" << std::hex << std::setfill('0');
    for(auto x:ep.commAddr.eid) s << std::setw(2) << unsigned(x);
    return s.str();
}
struct Path { CommLink link{}; uint32_t layer=0, die=0, remoteDie=0; std::string key; };
std::vector<Path> Discover(HcclComm comm, const Options& o) {
    uint32_t *raw=nullptr, n=0;
    CHECK(HcclRankGraphGetLayers(comm,&raw,&n));
    std::vector<uint32_t> layers;
    if(n) { if(!raw) throw std::runtime_error("null layers"); layers.assign(raw,raw+n); }
    std::vector<Path> paths;
    for(auto layer:layers) {
        CommTopo topo=COMM_TOPO_RESERVED;
        auto ts=HcclRankGraphGetTopoTypeByLayer(comm,layer,&topo);
        CommLink* links=nullptr; uint32_t count=0;
        auto status=HcclRankGraphGetLinks(comm,layer,o.rank,1-o.rank,&links,&count);
        std::cout << "L0L1_LAYER rank=" << o.rank << " layer=" << layer << " topo=" << int(topo)
                  << " topo_status=" << ts << " links_status=" << status << " count=" << count << std::endl;
        if(status!=HCCL_SUCCESS) continue;
        std::vector<CommLink> owned;
        if(count) { if(!links) throw std::runtime_error("null links"); owned.assign(links,links+count); }
        for(auto link:owned) {
            const bool eligible=link.linkAttr.linkProtocol==COMM_PROTOCOL_UBC_CTP &&
                link.srcEndpointDesc.commAddr.type==COMM_ADDR_TYPE_EID &&
                link.dstEndpointDesc.commAddr.type==COMM_ADDR_TYPE_EID &&
                ((layer==0 && link.linkAttr.hop==1) ||
                 (layer==1 && ts==HCCL_SUCCESS && topo==COMM_TOPO_CLOS));
            Path p; p.link=link; p.layer=layer;
            auto ds=HcclRankGraphGetEndpointInfo(comm,o.rank,&link.srcEndpointDesc,
                ENDPOINT_ATTR_DIE_ID,sizeof(p.die),&p.die);
            auto rs=HcclRankGraphGetEndpointInfo(comm,1-o.rank,&link.dstEndpointDesc,
                ENDPOINT_ATTR_DIE_ID,sizeof(p.remoteDie),&p.remoteDie);
            p.key=l0l1::PairKey(o.rank,EndpointKey(link.srcEndpointDesc),EndpointKey(link.dstEndpointDesc));
            bool usable=eligible && ds==HCCL_SUCCESS && rs==HCCL_SUCCESS && p.die<2 && p.remoteDie<2;
            std::cout << "L0L1_CANDIDATE rank=" << o.rank << " layer=" << layer
                      << " hop=" << unsigned(link.linkAttr.hop) << " protocol=" << int(link.linkAttr.linkProtocol)
                      << " die=" << p.die << " remote_die=" << p.remoteDie << " die_status=" << ds
                      << " remote_die_status=" << rs << " eligible=" << usable << " key=" << p.key << std::endl;
            if(usable) paths.push_back(p);
        }
    }
    std::sort(paths.begin(),paths.end(),[](const Path& a,const Path& b){
        return a.layer!=b.layer ? a.layer<b.layer : a.key<b.key;
    });
    paths.erase(std::unique(paths.begin(),paths.end(),[](const Path& a,const Path& b){
        return a.layer==b.layer && a.key==b.key;
    }),paths.end());
    unsigned indexes[2]={0,0};
    for(const auto& p:paths)
        std::cout << "L0L1_CATALOG rank=" << o.rank << " layer=" << p.layer << " index=" << indexes[p.layer]++
                  << " die=" << p.die << " key=" << p.key << std::endl;
    return paths;
}
Path Select(const std::vector<Path>& paths,unsigned layer,unsigned index) {
    for(const auto& p:paths) if(p.layer==layer) { if(!index--) return p; }
    throw std::runtime_error("PATH_UNAVAILABLE layer="+std::to_string(layer)+"; no fallback or fabricated EID");
}
struct Lane { Path path; aclrtStream stream=nullptr; ThreadHandle thread=0; ChannelHandle channel=0; CcuKernelHandle kernel=0; };
// A separate kernel per selected path/die. Positive size only: no CCU compare,
// branch, dynamic mask, synthetic relay or intermediate NPU memory.
CcuResult Exchange(CcuKernelArg opaque) {
    try {
        const auto channel=*static_cast<const ChannelHandle*>(opaque);
        ccu::Variable input,output,inToken,outToken,bytes;
        EMIT(ccu::LoadArg(input,0)); EMIT(ccu::LoadArg(output,1));
        EMIT(ccu::LoadArg(inToken,2)); EMIT(ccu::LoadArg(outToken,3)); EMIT(ccu::LoadArg(bytes,4));
        auto remoteOutput=ccu::GetResByChannel<ccu::Variable>(channel,0);
        auto remoteToken=ccu::GetResByChannel<ccu::Variable>(channel,1);
        EMIT(ccu::WriteVariableWithNotify(channel,output,0,0));
        EMIT(ccu::WriteVariableWithNotify(channel,outToken,1,1));
        EMIT(ccu::NotifyWait(channel,0)); EMIT(ccu::NotifyWait(channel,1));
        ccu::LocalAddr src; src.addr=input; src.token=inToken;
        ccu::RemoteAddr dst; dst.addr=remoteOutput; dst.token=remoteToken;
        ccu::Event done;
        EMIT(ccu::Write(channel,dst,src,bytes,done)); EMIT(ccu::EventWait(done));
        EMIT(ccu::NotifyRecord(channel,2)); EMIT(ccu::NotifyWait(channel,2));
        return CcuResult::CCU_SUCCESS;
    } catch(const AscendC::ccu::detail::CcuException& e) {
        std::cerr << "L0L1_BUILD_FAILED " << e.what() << std::endl; return e.code();
    } catch(const std::exception& e) {
        std::cerr << "L0L1_BUILD_FAILED " << e.what() << std::endl; return CcuResult::CCU_E_INTERNAL;
    }
}
void Libraries() {
    if(!HcommCcuKernelRegisterStart || !HcommCcuKernelRegister || !HcommCcuKernelRegisterEnd || !HcommCcuKernelLaunch)
        throw std::runtime_error("public CCU entry points unavailable");
    Dl_info info{};
    if(!dladdr(reinterpret_cast<void*>(HcommCcuKernelLaunch),&info) || !info.dli_fname)
        throw std::runtime_error("cannot identify loaded libhcomm");
    std::cout << "L0L1_LIBRARY symbol=HcommCcuKernelLaunch file=" << info.dli_fname << std::endl;
}
void Run(const Options& o) {
    Libraries(); CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(o.device));
    uint32_t devices=0; CHECK(aclrtGetDeviceCount(&devices));
    if(devices!=2) throw std::runtime_error("exactly two visible devices required");
    std::ostringstream config;
    config << o.bytes << ':' << o.mode << ':' << o.suite << ':' << o.warmup << ':' << o.iters << ':'
           << o.rounds << ':' << o.l0Percent << ':' << o.l0Index << ':' << o.l1Index << ':' << o.timing;
    Agree(o,"config",config.str());
    Phase(o,"comm_init"); HcclRootInfo root{};
    if(!o.rank) { CHECK(HcclGetRootInfo(&root)); Save(o.dir+"/root.bin",std::string(reinterpret_cast<char*>(&root),sizeof(root))); }
    else {
        const auto data=ReadWait(o.dir+"/root.bin");
        if(data.size()!=sizeof(root)) throw std::runtime_error("rootinfo size mismatch");
        std::copy(data.begin(),data.end(),reinterpret_cast<char*>(&root));
    }
    HcclComm comm=nullptr; CHECK(HcclCommInitRootInfo(2,&root,o.rank,&comm));
    Phase(o,"discover"); auto paths=Discover(comm,o);
    if(o.suite=="discover") {
        Agree(o,"discovery_complete"); CHECK(HcclCommDestroy(comm)); CHECK(aclrtResetDevice(o.device)); CHECK(aclFinalize());
        std::cout << "L0L1_RANK_PASS rank=" << o.rank << " suite=discover" << std::endl; return;
    }
    std::vector<Lane> lanes;
    for(unsigned layer=0;layer<2;++layer) {
        if((o.mode=="l0" && layer==1)||(o.mode=="l1" && layer==0)) continue;
        Lane lane; lane.path=Select(paths,layer,layer==0?o.l0Index:o.l1Index); lanes.push_back(lane);
    }
    if(lanes.size()==2 && lanes[0].path.key==lanes[1].path.key)
        throw std::runtime_error("PATH_ALIAS: L0 and L1 have identical endpoint pairs");
    for(auto& l:lanes) {
        const auto& p=l.path;
        const auto signature=p.key+":"+std::to_string(p.link.linkAttr.hop)+":"+
            std::to_string(o.rank?p.remoteDie:p.die)+":"+std::to_string(o.rank?p.die:p.remoteDie);
        Agree(o,"selected_L"+std::to_string(p.layer),signature);
    }
    Phase(o,"query_instance"); CcuInsHandle instance=0; uint32_t count=1;
    CHECK(HcclCommQueryCcuIns(comm,&instance,&count));
    if(!instance || count!=1) throw std::runtime_error("expected one borrowed CCU instance");
    // Stream handles select submission queues; kernel registration supplies die.
    for(auto& l:lanes) {
        CHECK(aclrtCreateStream(&l.stream));
        CHECK(HcclThreadAcquireWithStream(comm,COMM_ENGINE_CCU,l.stream,3,&l.thread));
    }
    void *send=nullptr,*recv=nullptr; const uint64_t capacity=o.bytes+2*l0l1::Guard;
    CHECK(aclrtMalloc(&send,capacity,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&recv,capacity,ACL_MEM_MALLOC_HUGE_FIRST));
    CommMem im{COMM_MEM_TYPE_DEVICE,send,capacity},om{COMM_MEM_TYPE_DEVICE,recv,capacity};
    HcclMemHandle regs[2]{};
    CHECK(HcclCommMemReg(comm,"l0l1_input",&im,&regs[0])); CHECK(HcclCommMemReg(comm,"l0l1_output",&om,&regs[1]));
    Phase(o,"acquire_channels");
    std::vector<HcclChannelDesc> descs(lanes.size()); std::vector<ChannelHandle> channels(lanes.size());
    CHECK(HcclChannelDescInit(descs.data(),descs.size()));
    for(size_t i=0;i<lanes.size();++i) {
        auto& d=descs[i]; const auto& p=lanes[i].path;
        d.remoteRank=1-o.rank; d.notifyNum=3; d.channelProtocol=p.link.linkAttr.linkProtocol;
        d.localEndpoint=p.link.srcEndpointDesc; d.remoteEndpoint=p.link.dstEndpointDesc;
        d.memHandles=regs; d.memHandleNum=2;
    }
    CHECK(HcclChannelAcquire(comm,COMM_ENGINE_CCU,descs.data(),descs.size(),channels.data()));
    if(channels.size()==2 && channels[0]==channels[1]) throw std::runtime_error("CHANNEL_ALIAS: identical handles");
    std::vector<int32_t> states(lanes.size(),-1);
    CHECK(HcommChannelGetStatus(channels.data(),channels.size(),states.data()));
    for(size_t i=0;i<lanes.size();++i) {
        auto& l=lanes[i]; l.channel=channels[i];
        std::cout << "L0L1_SELECTED rank=" << o.rank << " layer=" << l.path.layer << " die=" << l.path.die
                  << " remote_die=" << l.path.remoteDie << " hop=" << unsigned(l.path.link.linkAttr.hop)
                  << " channel=" << l.channel << " state=" << states[i] << " key=" << l.path.key << std::endl;
        if(states[i]!=0) throw std::runtime_error("channel not ready");
    }
    Phase(o,"register_start"); CHECK(HcommCcuKernelRegisterStart(instance));
    for(auto& l:lanes) {
        Phase(o,"register_kernel_L"+std::to_string(l.path.layer));
        const void* args[]={&l.channel}; const auto name="L0L1PositiveWriteL"+std::to_string(l.path.layer);
        CHECK(HcommCcuKernelRegister(instance,l.path.die,name.c_str(),reinterpret_cast<const void*>(Exchange),args,1,&l.kernel));
    }
    Phase(o,"register_end"); CHECK(HcommCcuKernelRegisterEnd(instance));
    for(const auto& l:lanes) { uint32_t args=0; CHECK(HcommCcuGetTaskArgsNum(l.kernel,&args));
        if(args!=5) throw std::runtime_error("unexpected task argument count"); }
    uint64_t inToken=0,outToken=0;
    CHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(send),capacity,&inToken));
    CHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(recv),capacity,&outToken));
    std::vector<uint32_t> input(capacity/4),output(capacity/4);
    constexpr uint32_t sentinel=0xfedcba98;
    auto sync=[&] { for(const auto& l:lanes) CHECK(aclrtSynchronizeStream(l.stream)); };
    auto launch=[&](const std::string& mode) {
        auto split=l0l1::Split(o.bytes,o.l0Percent);
        // Submit BOTH before waiting; separate streams even for same-die paths.
        for(const auto& l:lanes) {
            if(mode!="split" && mode!="l"+std::to_string(l.path.layer)) continue;
            const auto part=mode=="split"?split[l.path.layer]:l0l1::Slice{0,o.bytes};
            uint64_t args[]={reinterpret_cast<uint64_t>(send)+l0l1::Guard+part.offset,
                reinterpret_cast<uint64_t>(recv)+l0l1::Guard+part.offset,inToken,outToken,part.bytes};
            CHECK(HcommCcuKernelLaunch(l.thread,l.kernel,args,5));
        }
    };
    auto fill=[&](unsigned epoch) {
        std::fill(input.begin(),input.end(),sentinel); std::fill(output.begin(),output.end(),sentinel);
        for(uint64_t i=0;i<o.bytes/4;++i) input[l0l1::Guard/4+i]=l0l1::Pattern(o.rank,epoch,i);
        CHECK(aclrtMemcpy(send,capacity,input.data(),capacity,ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK(aclrtMemcpy(recv,capacity,output.data(),capacity,ACL_MEMCPY_HOST_TO_DEVICE));
    };
    auto verify=[&](unsigned epoch) {
        CHECK(aclrtMemcpy(output.data(),capacity,recv,capacity,ACL_MEMCPY_DEVICE_TO_HOST));
        for(uint64_t i=0;i<capacity/4;++i) {
            uint32_t expected=sentinel;
            if(i>=l0l1::Guard/4 && i<(l0l1::Guard+o.bytes)/4) expected=l0l1::Pattern(1-o.rank,epoch,i-l0l1::Guard/4);
            if(output[i]!=expected) throw std::runtime_error("payload/guard mismatch word="+std::to_string(i));
        }
    };
    std::vector<std::string> modes=o.mode=="all"?std::vector<std::string>{"l0","l1","split"}:std::vector<std::string>{o.mode};
    if(o.suite!="create") {
        Phase(o,"correctness");
        for(auto& mode:modes) for(unsigned epoch:{17U,31U,43U}) {
            fill(epoch); const auto tag="correct_"+mode+std::to_string(epoch);
            Agree(o,tag+"_ready"); launch(mode); sync(); verify(epoch); Agree(o,tag+"_checked");
            std::cout << "L0L1_CORRECT rank=" << o.rank << " mode=" << mode << " epoch=" << epoch << std::endl;
        }
    }
    if(o.suite=="benchmark" || o.suite=="all") {
        std::ofstream results(o.dir+"/rank"+std::to_string(o.rank)+".jsonl");
        if(!results) throw std::runtime_error("cannot open results");
        const unsigned orders[6][3]={{0,1,2},{1,2,0},{2,0,1},{0,2,1},{2,1,0},{1,0,2}};
        for(unsigned round=0;round<o.rounds;++round) for(unsigned slot=0;slot<modes.size();++slot) {
            const auto mode=modes[modes.size()==3?orders[round%6][slot]:0];
            const unsigned epoch=100+round*3+slot; fill(epoch);
            const auto tag="bench_"+std::to_string(round)+"_"+mode;
            Agree(o,tag+"_warmup");
            for(unsigned i=0;i<o.warmup;++i) launch(mode);
            sync(); verify(epoch); Agree(o,tag+"_measure");
            const auto begin=Clock::now();
            for(unsigned i=0;i<o.iters;++i) { launch(mode); if(o.timing=="serial") sync(); }
            const auto enqueued=Clock::now(); sync(); const auto done=Clock::now();
            verify(epoch); Agree(o,tag+"_checked");
            const double wall=std::chrono::duration<double,std::micro>(done-begin).count()/o.iters;
            const double host=std::chrono::duration<double,std::micro>(enqueued-begin).count()/o.iters;
            const auto split=l0l1::Split(o.bytes,o.l0Percent);
            const auto b0=mode=="l0"?o.bytes:mode=="l1"?0:split[0].bytes;
            const auto b1=o.bytes-b0;
            std::ostringstream row; row<<std::setprecision(12)<<"{\"rank\":"<<o.rank<<",\"round\":"<<round
                <<",\"mode\":\""<<mode<<"\",\"timing\":\""<<o.timing<<"\",\"bytes_per_direction\":"<<o.bytes
                <<",\"l0_bytes\":"<<b0<<",\"l1_bytes\":"<<b1<<",\"wall_us\":"<<wall
                <<",\"enqueue_or_serial_us\":"<<host<<"}";
            results<<row.str()<<std::endl; std::cout<<"L0L1_SAMPLE "<<row.str()<<std::endl;
        }
    }
    Phase(o,"teardown"); sync(); Agree(o,"teardown");
    // Comm owns instance/channels/registered memories; streams and buffers stay
    // alive through comm destruction. No unregister/destroy of borrowed instance.
    CHECK(HcclCommDestroy(comm)); CHECK(aclrtFree(send)); CHECK(aclrtFree(recv));
    for(auto& l:lanes) CHECK(aclrtDestroyStream(l.stream));
    CHECK(aclrtResetDevice(o.device)); CHECK(aclFinalize());
    std::cout << "L0L1_RANK_PASS rank=" << o.rank << " suite=" << o.suite << std::endl;
}
} // namespace
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--check-abi") { Libraries(); std::puts("L0L1_ABI_PASS"); return 0; }
        Run(Parse(argc,argv)); return 0;
    } catch(const std::exception& e) {
        std::cerr<<"L0L1_FAILED error="<<e.what()<<std::endl;
        // Do not run unsafe cleanup on a failed/asynchronous communicator.
        std::fflush(nullptr); std::_Exit(1);
    }
}
