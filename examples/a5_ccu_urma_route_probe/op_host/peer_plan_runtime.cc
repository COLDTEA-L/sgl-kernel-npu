#include "a5_ccu_urma_route_probe.h"
#include "utils.h"
#include "data_type.h"
#include "self_copy_lane.h"
#include "route_kernel.h"
#include "alltoallv_kernel.h"
#include <hcomm/ccu/hccl_ccu_res.h>
#include <hcomm/hcomm_res.h>
#include <atomic>
#include <fstream>
#include <mutex>
#include <map>
#include <sstream>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <iterator>

using namespace a5_ccu_urma_probe;
namespace {
struct PeerPreparedPlan {
    HcclComm comm = nullptr;
    int32_t device = -1;
    uint32_t rankSize = 0;
    bool allToAllV = false;
    std::string id, content;
    uint32_t availableCards = 0;
    ChannelSyncMode syncMode = ChannelSyncMode::BASELINE;
    std::vector<PeerPathSpec> specs;
    std::map<uintptr_t, PeerPlanResources> streams;
};
std::mutex registryMutex, buildMutex, submitMutex;
std::map<uint64_t, PeerPreparedPlan> plans;
std::atomic<uint64_t> nextHandle{1};

bool Uint(const std::string &text, uint32_t *value)
{
    if (text.empty() || text[0] == '-') return false;
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno || end == text.c_str() || *end != '\0' || parsed > UINT32_MAX) return false;
    *value = static_cast<uint32_t>(parsed);
    return true;
}
std::vector<std::string> Fields(const std::string &line)
{
    std::vector<std::string> fields;
    std::stringstream input(line);
    std::string value;
    while (std::getline(input, value, '\t')) fields.push_back(value);
    return fields;
}
HcclResult Load(const char *path, uint32_t rankSize, uint32_t availableCards,
    std::vector<PeerPathSpec> *specs, std::string *content)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() > 1024 * 1024) return HCCL_E_PARA;
    file.seekg(0);
    content->assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (file.bad()) return HCCL_E_PARA;
    std::stringstream input(*content);
    std::string line;
    std::getline(input, line);
    const auto header = Fields(line);
    std::map<std::string, size_t> columns;
    for (size_t i = 0; i < header.size(); ++i) {
        if (!columns.emplace(header[i], i).second) return HCCL_E_PARA;
    }
    for (const char *name : {"src_rank", "dst_rank", "src_phy", "dst_phy", "kind",
        "relay_phy", "direct_route", "weight", "src_die", "dst_die", "src_eid", "dst_eid"}) {
        if (!columns.count(name)) return HCCL_E_PARA;
    }
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto fields = Fields(line);
        if (fields.size() != header.size()) return HCCL_E_PARA;
        auto get = [&](const char *name) -> const std::string & { return fields[columns.at(name)]; };
        PeerPathSpec p;
        if (!Uint(get("src_rank"), &p.srcRank) || !Uint(get("dst_rank"), &p.dstRank) ||
            !Uint(get("src_phy"), &p.srcPhy) || !Uint(get("dst_phy"), &p.dstPhy) ||
            !Uint(get("direct_route"), &p.directRoute) || !Uint(get("weight"), &p.weight) ||
            !Uint(get("src_die"), &p.srcDie) || !Uint(get("dst_die"), &p.dstDie)) return HCCL_E_PARA;
        if (get("kind") == "relay") {
            p.relay = true;
            if (!Uint(get("relay_phy"), &p.relayPhy)) return HCCL_E_PARA;
            p.srcEid = get("src_eid"); p.dstEid = get("dst_eid");
        } else if (get("kind") != "direct") return HCCL_E_PARA;
        specs->push_back(p);
    }
    std::string error;
    if (!ValidatePeerPlan(rankSize, availableCards, *specs, &error)) {
        std::fprintf(stderr, "PEER_PLAN_INVALID %s\n", error.c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
HcclResult Bind(uint64_t handle, HcclComm comm, aclrtStream stream)
{
    if (handle == 0 || stream == nullptr) return HCCL_E_PARA;
    ChannelSyncMode requested;
    if (!ReadChannelSyncMode(&requested)) return HCCL_E_PARA;
    std::lock_guard<std::mutex> build(buildMutex);
    PeerPreparedPlan snapshot;
    {
        std::lock_guard<std::mutex> lock(registryMutex);
        const auto found = plans.find(handle);
        int32_t device = -1;
        if (found == plans.end()) return HCCL_E_NOT_FOUND;
        // The registered notification protocol is immutable for this plan,
        // including additional graph streams. Never silently reuse another mode.
        if (requested != found->second.syncMode) return HCCL_E_PARA;
        if ((comm && comm != found->second.comm) || aclrtGetDevice(&device) != ACL_SUCCESS ||
            device != found->second.device) return HCCL_E_PARA;
        if (found->second.streams.count(reinterpret_cast<uintptr_t>(stream))) return HCCL_SUCCESS;
        snapshot = found->second;
    }
    PeerPlanResources resources;
    const auto status = GetPeerPlanResources(snapshot.comm, stream, snapshot.specs, &resources, snapshot.syncMode, snapshot.allToAllV);
    if (status != HCCL_SUCCESS) return status;
    if (!resources.route.ccuSelfCopy) {
        const auto copyStatus = SelfCopyLane::Create(stream, &resources.route.selfCopy);
        if (copyStatus != HCCL_SUCCESS) return copyStatus;
    }
    {
        std::lock_guard<std::mutex> lock(registryMutex);
        plans.at(handle).streams.emplace(reinterpret_cast<uintptr_t>(stream), resources);
    }
    std::printf("PEER_PLAN_BOUND handle=%lu rank=%u ranks=%u channels=%zu stream=%p die=%u\n",
        static_cast<unsigned long>(handle), resources.route.rank, resources.route.rankSize,
        resources.peers.size(), stream, resources.route.dieId);
    std::fflush(stdout);
    std::printf("CCU_LOCAL_REMOTE_PLAN handle=%lu rank=%u enabled=%u self_copy=%s\n",
        static_cast<unsigned long>(handle), resources.route.rank, resources.route.ccuSelfCopy ? 1U : 0U,
        resources.route.ccuGroupCopy ? "GroupCopy8x32K" : resources.route.ccuSelfCopy ? "LocalCopyNb" : "ACL_D2D");
    std::fflush(stdout);
    return HCCL_SUCCESS;
}
} // namespace

extern "C" __attribute__((visibility("default"))) int A5CcuPeerPlanAbiVersion() { return 5; }
extern "C" __attribute__((visibility("default"))) int A5CcuChannelSyncAbiVersion() { return 1; }

static HcclResult CreatePeerPlan(HcclComm comm, aclrtStream stream,
    const char *planId, const char *manifest, uint32_t availableCards, uint64_t *handle, bool allToAllV)
{
    if (!comm || !stream || !planId || !manifest || !handle) return HCCL_E_PTR;
    *handle = 0;
    PeerPreparedPlan plan;
    if (!ReadChannelSyncMode(&plan.syncMode)) return HCCL_E_PARA;
    if (!*planId || HcclGetRankSize(comm, &plan.rankSize) != HCCL_SUCCESS ||
        aclrtGetDevice(&plan.device) != ACL_SUCCESS) return HCCL_E_PARA;
    if (allToAllV ? (plan.rankSize != 8 || plan.syncMode != ChannelSyncMode::BASELINE) :
        (plan.rankSize != 2 && plan.rankSize != 4)) return HCCL_E_PARA;
    plan.allToAllV = allToAllV;
    plan.comm = comm; plan.id = planId; plan.availableCards = availableCards;
    const auto status = Load(manifest, plan.rankSize, availableCards, &plan.specs, &plan.content);
    if (status != HCCL_SUCCESS) return status;
    uint64_t id = 0;
    bool inserted = false;
    {
        std::lock_guard<std::mutex> lock(registryMutex);
        for (const auto &entry : plans) {
            const auto &p = entry.second;
            if (p.comm == comm && p.id == plan.id) {
                if (p.content != plan.content || p.device != plan.device || p.availableCards != availableCards ||
                    p.syncMode != plan.syncMode || p.allToAllV != plan.allToAllV)
                    return HCCL_E_PARA;
                id = entry.first;
                break;
            }
        }
        if (id == 0) {
            id = nextHandle.fetch_add(1);
            plans.emplace(id, plan);
            inserted = true;
        }
    }
    const auto bound = Bind(id, comm, stream);
    if (bound != HCCL_SUCCESS) {
        if (inserted) { std::lock_guard<std::mutex> lock(registryMutex); plans.erase(id); }
        return bound;
    }
    *handle = id;
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclCcuUrmaPeerPlanCreate(HcclComm comm, aclrtStream stream,
    const char *planId, const char *manifest, uint32_t availableCards, uint64_t *handle)
{ return CreatePeerPlan(comm, stream, planId, manifest, availableCards, handle, false); }

extern "C" int A5CcuAllToAllVAbiVersion() { return 1; }
extern "C" HcclResult HcclCcuUrmaPeerPlanAllToAllVCreate(HcclComm comm, aclrtStream stream,
    const char *planId, const char *manifest, uint32_t availableCards, uint64_t *handle)
{ return CreatePeerPlan(comm, stream, planId, manifest, availableCards, handle, true); }

extern "C" HcclResult HcclCcuUrmaPeerPlanBindStream(uint64_t handle, HcclComm comm, aclrtStream stream)
{
    return Bind(handle, comm, stream);
}

extern "C" HcclResult HcclCcuUrmaPeerPlanExecute(void *send, void *recv,
    uint64_t elementsPerPeer, uint32_t rankSize, HcclDataType dtype, HcclComm comm, aclrtStream stream,
    uint64_t handle, const uint32_t *policy, uint32_t policyCount)
{
    if (!send || !recv || !stream) return HCCL_E_PTR;
    const uint32_t elementBytes = PayloadElementBytes(dtype);
    if (!elementBytes || (policyCount && !policy)) return HCCL_E_PARA;
    std::lock_guard<std::mutex> submit(submitMutex);
    PeerPlanResources resources;
    HcclComm owner;
    {
        std::lock_guard<std::mutex> lock(registryMutex);
        const auto found = plans.find(handle);
        int32_t device = -1;
        if (found == plans.end()) return HCCL_E_NOT_FOUND;
        if ((comm && comm != found->second.comm) || aclrtGetDevice(&device) != ACL_SUCCESS ||
            device != found->second.device) return HCCL_E_PARA;
        const auto bound = found->second.streams.find(reinterpret_cast<uintptr_t>(stream));
        if (bound == found->second.streams.end()) {
            std::fprintf(stderr, "PEER_PLAN_UNBOUND bind execution/capture stream outside graph first\n");
            return HCCL_E_NOT_FOUND;
        }
        if (found->second.allToAllV) return HCCL_E_PARA;
        owner = found->second.comm;
        resources = bound->second;
    }
    auto &r = resources.route;
    if (rankSize != r.rankSize) return HCCL_E_PARA;
    std::vector<uint32_t> weights = r.weights;
    if (policyCount) {
        if (policyCount != weights.size()) return HCCL_E_PARA;
        weights.assign(policy, policy + policyCount);
    }
    PeerPathLayout layout;
    if (!BuildPeerPathLayout(elementsPerPeer, r.rank, r.rankSize, resources.peers, weights, &layout, elementBytes) ||
        !DisjointBufferRanges(reinterpret_cast<uint64_t>(send), reinterpret_cast<uint64_t>(recv),
                              layout.totalBytes)) return HCCL_E_PARA;
    if (r.ccuGroupCopy && layout.peerBytes > GROUP_MAX_BYTES) return HCCL_E_PARA;
    const uint64_t inputToken = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(send), layout.totalBytes);
    const uint64_t outputToken = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(recv), layout.totalBytes);
    auto *input = static_cast<unsigned char *>(send);
    auto *output = static_cast<unsigned char *>(recv);
    SelfCopyGuard copy(r.selfCopy);
    if (!r.ccuSelfCopy) {
        const auto copyStatus = copy.Begin(output + layout.selfOffset, input + layout.selfOffset, layout.peerBytes);
        if (copyStatus != HCCL_SUCCESS) return copyStatus;
    }
    auto status = HCCL_SUCCESS;
    if (r.routeThread != r.mainThread) {
        status = static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(r.mainThread, r.routeThread, 0));
        if (status != HCCL_SUCCESS) return status;
        status = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(r.routeThread, 0, 1800));
        if (status != HCCL_SUCCESS) return status;
    }
    RouteTaskArg task(reinterpret_cast<uint64_t>(send), reinterpret_cast<uint64_t>(recv),
        inputToken, outputToken, layout.sourceOffsets, layout.remoteOffsets, layout.pathBytes,
        layout.selfOffset, layout.peerBytes);
    status = HcclCcuKernelLaunch(owner, r.routeThread, r.kernel, &task);
    if (status != HCCL_SUCCESS || r.routeThread == r.mainThread) return copy.Finish(status);
    status = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(r.mainThread, 0, 1800));
    if (status != HCCL_SUCCESS) return status;
    return copy.Finish(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(r.routeThread, r.mainThread, 0)));
}

extern "C" HcclResult HcclCcuUrmaPeerPlanAllToAllVExecute(void *send, void *recv,
    uint64_t sendCapacity, uint64_t recvCapacity, uint64_t rowBytes, uint32_t rank,
    HcclComm comm, aclrtStream stream, uint64_t handle, const uint64_t *counts, uint32_t count)
{
    if (!send || !recv || !stream || !counts) return HCCL_E_PTR;
    std::lock_guard<std::mutex> submit(submitMutex);
    PeerPlanResources resources;
    HcclComm owner;
    {
        std::lock_guard<std::mutex> lock(registryMutex);
        const auto found = plans.find(handle);
        int32_t device = -1;
        if (found == plans.end()) return HCCL_E_NOT_FOUND;
        if (!found->second.allToAllV || (comm && comm != found->second.comm) ||
            aclrtGetDevice(&device) != ACL_SUCCESS || device != found->second.device) return HCCL_E_PARA;
        const auto bound = found->second.streams.find(reinterpret_cast<uintptr_t>(stream));
        if (bound == found->second.streams.end()) return HCCL_E_NOT_FOUND;
        resources = bound->second; owner = found->second.comm;
    }
    const auto &r = resources.route;
    AllToAllVLayout layout;
    if (rank != r.rank || r.rankSize != 8 ||
        !BuildAllToAllVLayout(rank, counts, count, rowBytes, resources.peers, &layout) ||
        !DisjointAllToAllVBuffers(reinterpret_cast<uint64_t>(send), sendCapacity,
            reinterpret_cast<uint64_t>(recv), recvCapacity, layout) || layout.selfBytes > GROUP_MAX_BYTES)
        return HCCL_E_PARA;
    const uint64_t inputToken = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(send), sendCapacity);
    const uint64_t outputToken = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(recv), recvCapacity);
    AllToAllVTaskArg task(reinterpret_cast<uint64_t>(send), reinterpret_cast<uint64_t>(recv),
        inputToken, outputToken, layout);
    auto status = HCCL_SUCCESS;
    if (r.routeThread != r.mainThread) {
        status = static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(r.mainThread, r.routeThread, 0));
        if (status != HCCL_SUCCESS) return status;
        status = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(r.routeThread, 0, 1800));
        if (status != HCCL_SUCCESS) return status;
    }
    status = HcclCcuKernelLaunch(owner, r.routeThread, r.kernel, &task);
    if (status != HCCL_SUCCESS || r.routeThread == r.mainThread) return status;
    status = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(r.mainThread, 0, 1800));
    if (status != HCCL_SUCCESS) return status;
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(r.routeThread, r.mainThread, 0));
}

extern "C" HcclResult HcclCcuUrmaPeerPlanDestroy(uint64_t handle)
{
    std::lock_guard<std::mutex> build(buildMutex);
    std::lock_guard<std::mutex> submit(submitMutex);
    std::lock_guard<std::mutex> lock(registryMutex);
    const auto found = plans.find(handle);
    if (found == plans.end()) return HCCL_E_NOT_FOUND;
    for (auto &stream : found->second.streams) {
        if (stream.second.route.selfCopy) {
            const auto status = stream.second.route.selfCopy->Close();
            if (status != HCCL_SUCCESS) return status;
        }
    }
    plans.erase(found);
    return HCCL_SUCCESS;
}
