// Compile the PRODUCTION RouteKernel against a CPU instruction recorder.
// Tests graph ordering/argument serialization and byte placement, not hardware.
#include "../../examples/a5_ccu_urma_route_probe/op_kernel_ccu/route_kernel.h"
#include "../../examples/a5_ccu_urma_route_probe/common/peer_plan.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <set>
using namespace a5_ccu_urma_probe;
class TestKernel : public RouteKernel {
public:
    using RouteKernel::RouteKernel;
    using RouteKernel::GeneArgs;
};
static void CheckGraph(const TestKernel &kernel, size_t paths, bool local,bool group=false,
    ChannelSyncMode mode=ChannelSyncMode::BASELINE)
{
    size_t locals=0, remotes=0, waits=0, postRecords=0, postWaits=0;
    std::set<int> events, waited;
    bool firstWait=false;
    size_t published=0, metadataWaits=0;
    std::set<ChannelHandle> ready;
    for (const auto &i : kernel.instructions) {
        if(i.op=="record" && i.slot!=2) {
            ++published;
            assert(metadataWaits==0 && i.variable<=1);
            if(mode==ChannelSyncMode::BASELINE) assert(i.slot==i.variable && i.mask==1);
            else assert(i.slot==0 && i.mask==(1U<<i.variable));
        }
        if(i.op=="notify_wait" && i.slot!=2) {
            assert(published==2*paths); ++metadataWaits;
            if(mode!=ChannelSyncMode::BASELINE) {
                assert(i.slot==0 && i.mask==3); ready.insert(i.channel);
                // Both bits required and consumed before the next epoch. This
                // models the assumed SDK semantics, not a hardware proof.
                uint32_t bits=1; assert((bits&i.mask)!=i.mask);
                bits|=2; assert((bits&i.mask)==i.mask);
                bits&=~i.mask; assert(bits==0);
            } else if(i.slot==1) ready.insert(i.channel);
        }
        if(i.op=="remote") {
            assert(ready.count(i.channel));
            if(mode==ChannelSyncMode::PACKED_READY) assert(metadataWaits==remotes+1);
            else assert(metadataWaits==(mode==ChannelSyncMode::BASELINE?2:1)*paths);
        }
        if(i.op=="group_local") { assert(group && !firstWait && remotes==paths); ++locals; }
        if (i.op=="local" || i.op=="remote") {
            assert(!firstWait); // all data submissions before any completion wait
            assert(events.insert(i.event).second);
            if (i.op=="local") ++locals; else ++remotes;
        }
        if (i.op=="event_wait") {
            firstWait=true; ++waits;
            assert(events.count(i.event)); assert(waited.insert(i.event).second);
        }
        if (i.slot==2 && i.op=="record") { assert(waits==paths+(local&&!group?1:0)); ++postRecords; }
        if (i.slot==2 && i.op=="notify_wait") { assert(postRecords==paths); ++postWaits; }
    }
    assert(locals==(local?1U:0U) && remotes==paths && waits==events.size());
    assert(postRecords==paths && postWaits==paths);
    assert(metadataWaits==(mode==ChannelSyncMode::BASELINE?2:1)*paths);
}
static void Run(uint32_t k, uint32_t pathsPerPeer, uint32_t elementBytes,
    size_t peerBytes=4096, uint32_t epochs=40, ChannelSyncMode mode=ChannelSyncMode::BASELINE)
{
    constexpr size_t guard=32;
    const size_t total=k*peerBytes;
    std::vector<std::vector<unsigned char>> input(k), output(k);
    std::vector<std::unique_ptr<TestKernel>> kernels;
    std::vector<PeerPathLayout> layouts(k);
    for (uint32_t rank=0; rank<k; ++rank) {
        std::vector<uint32_t> peers, weights, ordinals;
        std::vector<ChannelHandle> channels;
        for (uint32_t peer=0; peer<k; ++peer) if (rank!=peer) {
            for (uint32_t path=0; path<pathsPerPeer; ++path) {
                peers.push_back(peer); weights.push_back(path==0?2:1);
                ordinals.push_back(path); channels.push_back(1+peer*100+path);
            }
        }
        assert(BuildPeerPathLayout(peerBytes/elementBytes, rank, k, peers, weights, &layouts[rank], elementBytes));
        RouteKernelArg arg(channels, ordinals, true,true,mode), old(channels, ordinals, false);
        assert(arg.GetKernelSignature().value!=old.GetKernelSignature().value);
        kernels.emplace_back(new TestKernel(arg));
        assert(kernels.back()->Init()==0);
        CheckGraph(*kernels.back(), channels.size(), true,true,mode);
        input[rank].resize(total+2*guard); output[rank].resize(total+2*guard);
    }
    // Reuse registered graphs and buffers for changing epochs, rotate rank and
    // reverse copy completion order: no dependence on a particular submission.
    for (uint32_t epoch=0; epoch<epochs; ++epoch) {
        for (uint32_t rank=0; rank<k; ++rank) {
            std::fill(input[rank].begin(), input[rank].end(), 0xb7);
            std::fill(output[rank].begin(), output[rank].end(), 0xc3);
            for (uint32_t peer=0; peer<k; ++peer)
                for (size_t j=0; j<peerBytes; ++j)
                    input[rank][guard+peer*peerBytes+j]=(epoch*17+rank*31+peer*7+j)%256;
        }
        std::vector<std::pair<hcomm::Instruction, hcomm::CcuRep::Env>> copies;
        for (uint32_t n=0; n<k; ++n) {
            const uint32_t rank=(n+epoch)%k; const auto &l=layouts[rank];
            RouteTaskArg task(reinterpret_cast<uint64_t>(input[rank].data()+guard),
                reinterpret_cast<uint64_t>(output[rank].data()+guard), 1, 2,
                l.sourceOffsets,l.remoteOffsets,l.pathBytes,l.selfOffset,l.peerBytes);
            hcomm::CcuRep::Env env;
            env.args=kernels[rank]->GeneArgs(task);
            assert(env.args.size()==9+3*l.pathBytes.size());
            env.remote=[&output](ChannelHandle ch,uint32_t slot) {
                return slot==0 ? reinterpret_cast<uint64_t>(output[(ch-1)/100].data()+guard) : 2ULL;
            };
            for (const auto &i:kernels[rank]->instructions)
                if(i.op=="local" || i.op=="remote" || i.op=="group_local") copies.emplace_back(i,env);
        }
        if(epoch%2) std::reverse(copies.begin(),copies.end());
        for(const auto &copy:copies) {
            const auto &i=copy.first; const auto &e=copy.second;
            std::memcpy(reinterpret_cast<void *>(i.dst.eval(e)),
                reinterpret_cast<void *>(i.src.eval(e)),i.len.eval(e));
        }
        for (uint32_t dst=0;dst<k;++dst) {
            for (uint32_t peer=0;peer<k;++peer)
                for(size_t j=0;j<peerBytes;++j)
                    assert(input[dst][guard+peer*peerBytes+j]==(epoch*17+dst*31+peer*7+j)%256);
            for (uint32_t src=0;src<k;++src)
                assert(std::memcmp(output[dst].data()+guard+src*peerBytes,
                    input[src].data()+guard+dst*peerBytes,peerBytes)==0);
            for(size_t j=0;j<guard;++j) {
                assert(output[dst][j]==0xc3 && output[dst][guard+total+j]==0xc3);
                assert(input[dst][j]==0xb7 && input[dst][guard+total+j]==0xb7);
            }
        }
    }
}
int main()
{
    ChannelSyncMode parsed;
    unsetenv("A5_CCU_CHANNEL_SYNC_MODE");
    assert(ReadChannelSyncMode(&parsed) && parsed==ChannelSyncMode::BASELINE);
    for(auto mode:{ChannelSyncMode::BASELINE,ChannelSyncMode::PACKED,ChannelSyncMode::PACKED_READY}) {
        setenv("A5_CCU_CHANNEL_SYNC_MODE",ChannelSyncName(mode),1);
        assert(ReadChannelSyncMode(&parsed) && parsed==mode);
    }
    setenv("A5_CCU_CHANNEL_SYNC_MODE","typo",1); assert(!ReadChannelSyncMode(&parsed));
    unsetenv("A5_CCU_CHANNEL_SYNC_MODE");
    for(auto mode:{ChannelSyncMode::BASELINE,ChannelSyncMode::PACKED,ChannelSyncMode::PACKED_READY}) {
    for (auto bytes:{1U,2U,4U}) {
        for(auto paths:{1U,2U,3U,5U,7U}) Run(2,paths,bytes,4096,40,mode);
        for(auto paths:{1U,2U}) Run(4,paths,bytes,4096,40,mode);
    }
    for(auto bytes:{33024U,262400U,4194304U}) {
        Run(2,7,4,bytes,2,mode); Run(4,2,4,bytes,2,mode);
    }
    for (auto local:{false,true}) {
        RouteKernelArg arg({1,2},{0,1},local,false,mode); TestKernel kernel(arg);
        assert(kernel.Init()==0); CheckGraph(kernel,2,local,false,mode);
        RouteTaskArg task(1,2,3,4,{0,256},{0,256},{256,256},0,512);
        assert(kernel.GeneArgs(task).size()==(local?12U:10U));
        task.pathBytes.pop_back(); assert(kernel.GeneArgs(task).empty());
    }
    RouteKernelArg arg({1},{0},true); TestKernel kernel(arg);
    RouteTaskArg missing(1,2,3,4,{0},{0},{256}); assert(kernel.GeneArgs(missing).empty());
    for(auto local:{false,true}) {
        RouteKernelArg over(std::vector<ChannelHandle>(15,1),std::vector<uint32_t>(15,0),local);
        TestKernel overKernel(over); assert(overKernel.Init()==HCCL_E_PARA);
        assert(overKernel.instructions.empty());
    }
    RouteKernelArg boundary(std::vector<ChannelHandle>(13,1),std::vector<uint32_t>(13,0),true);
    TestKernel boundaryKernel(boundary); assert(boundaryKernel.Init()==HCCL_SUCCESS);
    RouteKernelArg tooMany(std::vector<ChannelHandle>(14,1),std::vector<uint32_t>(14,0),true);
    TestKernel tooManyKernel(tooMany); assert(tooManyKernel.Init()==HCCL_E_PARA);
    }
    std::set<std::string> signatures;
    for(auto mode:{ChannelSyncMode::BASELINE,ChannelSyncMode::PACKED,ChannelSyncMode::PACKED_READY}) {
        RouteKernelArg arg({1,2},{0,1},true,true,mode);
        assert(signatures.insert(arg.GetKernelSignature().value).second);
    }
    std::cout<<"CPU_MODEL_PASS: production combined graph, 2/4 ranks, byte sizes 1/2/4, disjoint rows/guards, event ordering; NOT hardware proof\n";
}
