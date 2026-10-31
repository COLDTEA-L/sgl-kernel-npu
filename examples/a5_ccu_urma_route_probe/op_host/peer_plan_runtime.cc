#include "a5_ccu_urma_route_probe.h"
#include "utils.h"
#include "data_type.h"
#include "self_copy_lane.h"
#include "thread_event_bridge.h"
#include "ep8_dfx.h"
#include <cstring>
#include "route_kernel.h"
#include "alltoallv_kernel.h"
#include "eid_alltoallv_kernel.h"
#include "../common/public_ccu.h"
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
    bool allToAllV = false, eidAllToAllV = false;
    EidRegisterStage eidStage = EidRegisterStage::FULL;
    std::shared_ptr<Ep8Dfx> dfx;
    std::string id, content;
    uint32_t availableCards = 0;
    ChannelSyncMode syncMode = ChannelSyncMode::BASELINE;
    std::vector<PeerPathSpec> specs;
    std::map<uintptr_t, PeerPlanResources> streams;
    std::map<uintptr_t, std::shared_ptr<ThreadEventBridge>> bridges;
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
    std::vector<PeerPathSpec> *specs, std::string *content, bool eidAllToAllV)
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
        else if (rankSize == 8 && (get("src_eid") != "-" || get("dst_eid") != "-")) {
            if (get("src_eid").empty() || get("dst_eid").empty() ||
                get("src_eid") == "-" || get("dst_eid") == "-") return HCCL_E_PARA;
            p.srcEid = get("src_eid"); p.dstEid = get("dst_eid");
        }
        specs->push_back(p);
    }
    std::string error;
    if (!ValidatePeerPlan(rankSize, availableCards, *specs, &error, eidAllToAllV)) {
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
    const auto status = GetPeerPlanResources(snapshot.comm, stream, snapshot.specs, &resources, snapshot.syncMode, snapshot.allToAllV, snapshot.eidAllToAllV, snapshot.eidStage);
    if (status != HCCL_SUCCESS) return status;
    std::shared_ptr<ThreadEventBridge> bridge;
    if (snapshot.allToAllV) {
        const char *mode = std::getenv("A5_CCU_EP8_THREAD_BRIDGE");
        const bool useEvent = !mode || std::strcmp(mode, "event") == 0;
        if (!useEvent && std::strcmp(mode, "hcomm") != 0) return HCCL_E_PARA;
        if (useEvent && resources.route.routeThread != resources.route.mainThread) {
            const auto bridgeStatus = ThreadEventBridge::Create(stream, resources.route.slaveStream, &bridge);
            if (bridgeStatus != HCCL_SUCCESS) return bridgeStatus;
        }
        std::printf("EP8_THREAD_BRIDGE rank=%u die=%u mode=%s main_thread=%lu route_thread=%lu main=%p route=%p events=%u\n",
            resources.route.rank, resources.route.dieId,
            resources.route.routeThread == resources.route.mainThread ? "same_thread" : useEvent ? "event" : "hcomm",
            resources.route.mainThread, resources.route.routeThread, stream, resources.route.slaveStream, bridge ? 2U : 0U);
    }
    if (!resources.route.ccuSelfCopy) {
        const auto copyStatus = SelfCopyLane::Create(stream, &resources.route.selfCopy);
        if (copyStatus != HCCL_SUCCESS) return copyStatus;
    }
    {
        std::lock_guard<std::mutex> lock(registryMutex);
        plans.at(handle).streams.emplace(reinterpret_cast<uintptr_t>(stream), resources);
        if (bridge) plans.at(handle).bridges.emplace(reinterpret_cast<uintptr_t>(stream), bridge);
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
    const char *planId, const char *manifest, uint32_t availableCards, uint64_t *handle, bool allToAllV, bool eidAllToAllV = false)
{
    if (!comm || !stream || !planId || !manifest || !handle) return HCCL_E_PTR;
    *handle = 0;
    PeerPreparedPlan plan;
    if (!ReadChannelSyncMode(&plan.syncMode)) return HCCL_E_PARA;
    if (!*planId || HcclGetRankSize(comm, &plan.rankSize) != HCCL_SUCCESS ||
        aclrtGetDevice(&plan.device) != ACL_SUCCESS) return HCCL_E_PARA;
    if (allToAllV ? (plan.rankSize != 8 || plan.syncMode != ChannelSyncMode::BASELINE) :
        (plan.rankSize != 2 && plan.rankSize != 4)) return HCCL_E_PARA;
    plan.allToAllV = allToAllV; plan.eidAllToAllV = eidAllToAllV;
    if (eidAllToAllV && !ReadEidRegisterStage(&plan.eidStage)) return HCCL_E_PARA;
    plan.comm = comm; plan.id = planId; plan.availableCards = availableCards;
    const auto status = Load(manifest, plan.rankSize, availableCards, &plan.specs, &plan.content, eidAllToAllV);
    if (status != HCCL_SUCCESS) return status;
    uint64_t id = 0;
    bool inserted = false;
    {
        std::lock_guard<std::mutex> lock(registryMutex);
        for (const auto &entry : plans) {
            const auto &p = entry.second;
            if (p.comm == comm && p.id == plan.id) {
                if (p.content != plan.content || p.device != plan.device || p.availableCards != availableCards ||
                    p.syncMode != plan.syncMode || p.allToAllV != plan.allToAllV || p.eidAllToAllV != plan.eidAllToAllV || p.eidStage != plan.eidStage)
                    return HCCL_E_PARA;
                id = entry.first;
                break;
            }
        }
        if (id == 0) {
            id = nextHandle.fetch_add(1);
            if (allToAllV) {
                const auto dfxStatus = Ep8Dfx::Create(comm, id, &plan.dfx);
                if (dfxStatus != HCCL_SUCCESS) return dfxStatus;
            }
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
// New opt-in manifest endpoint verification; legacy '-' direct rows are unchanged.
extern "C" int A5CcuBoundDirectAbiVersion() { return 1; }
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
    const uint64_t inputToken = a5ccu::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(send), layout.totalBytes);
    const uint64_t outputToken = a5ccu::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(recv), layout.totalBytes);
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
    status = a5ccu::LaunchKernel(owner, r.routeThread, r.kernel, &task);
    if (status != HCCL_SUCCESS || r.routeThread == r.mainThread) return copy.Finish(status);
    status = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(r.mainThread, 0, 1800));
    if (status != HCCL_SUCCESS) return status;
    return copy.Finish(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(r.routeThread, r.mainThread, 0)));
}

static HcclResult ExecuteAllToAllV(void *send, void *recv,
    uint64_t sendCapacity, uint64_t recvCapacity, uint64_t rowBytes, uint32_t rank,
    HcclComm comm, aclrtStream stream, uint64_t handle, const uint64_t *counts, uint32_t count,
    const uint64_t *storageCounts, uint32_t storageCount,
    const int32_t *relays = nullptr, const uint64_t *moved = nullptr)
{
    std::lock_guard<std::mutex> submit(submitMutex);
    auto checkedStatus = [&](HcclResult status, const char *stage) {
        if (status != HCCL_SUCCESS) {
            std::fprintf(stderr, "ALLTOALLV_STAGE_FAILED rank=%u handle=%lu stage=%s status=%d "
                "stream=%p send=%p recv=%p counts=%p count=%u send_capacity=%lu recv_capacity=%lu\n",
                rank, handle, stage, static_cast<int>(status), stream, send, recv,
                static_cast<const void *>(counts), count, sendCapacity, recvCapacity);
            std::fflush(stderr);
        }
        return status;
    };
    if (!send || !recv || !stream || !counts) return checkedStatus(HCCL_E_PTR, "entry_pointer");
    PeerPlanResources resources;
    std::shared_ptr<ThreadEventBridge> bridge;
    std::shared_ptr<Ep8Dfx> dfx;
    HcclComm owner;
    {
        std::lock_guard<std::mutex> lock(registryMutex);
        const auto found = plans.find(handle);
        int32_t device = -1;
        if (found == plans.end()) return checkedStatus(HCCL_E_NOT_FOUND, "lookup_plan_or_stream");
        if (!EidStageAllowsPayload(found->second.eidStage))
            return checkedStatus(HCCL_E_PARA, "registration_probe_cannot_execute");
        if (!found->second.allToAllV || found->second.eidAllToAllV != (relays != nullptr) || (comm && comm != found->second.comm) ||
            aclrtGetDevice(&device) != ACL_SUCCESS || device != found->second.device) return checkedStatus(HCCL_E_PARA, "validate_plan_device_layout");
        const auto bound = found->second.streams.find(reinterpret_cast<uintptr_t>(stream));
        if (bound == found->second.streams.end()) return checkedStatus(HCCL_E_NOT_FOUND, "lookup_plan_or_stream");
        resources = bound->second; owner = found->second.comm;
        dfx = found->second.dfx;
        const auto bridgeFound = found->second.bridges.find(reinterpret_cast<uintptr_t>(stream));
        if (bridgeFound != found->second.bridges.end()) bridge = bridgeFound->second;
    }
    const auto &r = resources.route;
    AllToAllVLayout layout;
    EidAllToAllVLayout eidLayout;
    bool validLayout = false;
    if (relays) {
        validLayout = !storageCounts && BuildEidAllToAllVLayout(rank, counts, count, rowBytes, relays, moved,
                          resources.eidAllowed, resources.eidChannels, &eidLayout);
        if (validLayout) layout = eidLayout;
    } else validLayout = BuildAllToAllVLayoutWithStorage(rank, counts, count, rowBytes, resources.peers,
                                                        storageCounts, storageCount, &layout);
    if (rank != r.rank || r.rankSize != 8 || !validLayout ||
        !DisjointAllToAllVBuffers(reinterpret_cast<uint64_t>(send), sendCapacity,
            reinterpret_cast<uint64_t>(recv), recvCapacity, layout) || layout.selfBytes > GROUP_MAX_BYTES)
        return checkedStatus(HCCL_E_PARA, "validate_plan_device_layout");
    const uint64_t inputToken = a5ccu::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(send), sendCapacity);
    const uint64_t outputToken = a5ccu::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(recv), recvCapacity);
    AllToAllVTaskArg directTask(reinterpret_cast<uint64_t>(send), reinterpret_cast<uint64_t>(recv),
        inputToken, outputToken, layout);
    EidAllToAllVTaskArg eidTask(reinterpret_cast<uint64_t>(send), reinterpret_cast<uint64_t>(recv),
        inputToken, outputToken, eidLayout);
    const a5ccu::CcuTaskArg* task = relays ? static_cast<const a5ccu::CcuTaskArg*>(&eidTask) : &directTask;
    if (!dfx) return checkedStatus(HCCL_E_PTR, "dfx_context");
    uint64_t payloadBytes = layout.selfBytes;
    for (auto bytes : layout.pathBytes) payloadBytes += bytes; // bounded by validated storage layout
    return dfx->Run(rank, reinterpret_cast<uint64_t>(send), reinterpret_cast<uint64_t>(recv),
        sendCapacity, recvCapacity, payloadBytes, [&]() -> HcclResult {
        auto status = HCCL_SUCCESS;
        if (bridge) {
            status = checkedStatus(bridge->Begin(), "event_route_start");
            if (status != HCCL_SUCCESS) return status;
        } else if (r.routeThread != r.mainThread) {
            status = checkedStatus(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(r.mainThread, r.routeThread, 0)), "record_route_start");
            if (status != HCCL_SUCCESS) return status;
            status = checkedStatus(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(r.routeThread, 0, 1800)), "wait_route_start");
            if (status != HCCL_SUCCESS) return status;
        }
        status = checkedStatus(a5ccu::LaunchKernel(owner, r.routeThread, r.kernel, task), "kernel_launch");
        if (bridge) {
            const auto finishStatus = bridge->Finish(status);
            // Finish preserves a failed launch and poisons the bridge; do not log
            // that same failure again as an event failure. Never retry a launch.
            return status != HCCL_SUCCESS ? status : checkedStatus(finishStatus, "event_route_done");
        }
        if (status != HCCL_SUCCESS || r.routeThread == r.mainThread) return status;
        status = checkedStatus(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(r.mainThread, 0, 1800)), "wait_route_done");
        if (status != HCCL_SUCCESS) return status;
        return checkedStatus(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(r.routeThread, r.mainThread, 0)), "record_route_done");
    });
}

// Additive ABI: the old contiguous interface remains unchanged.
extern "C" int A5CcuAllToAllVLayoutAbiVersion() { return 1; }
extern "C" int A5CcuEp8ThreadBridgeAbiVersion() { return 1; }
extern "C" int A5CcuEp8DfxAbiVersion() { return 2; }
extern "C" HcclResult HcclCcuUrmaPeerPlanAllToAllVExecute(void *send, void *recv,
    uint64_t sendCapacity, uint64_t recvCapacity, uint64_t rowBytes, uint32_t rank,
    HcclComm comm, aclrtStream stream, uint64_t handle, const uint64_t *counts, uint32_t count)
{
    return ExecuteAllToAllV(send, recv, sendCapacity, recvCapacity, rowBytes, rank,
        comm, stream, handle, counts, count, nullptr, 0);
}
extern "C" HcclResult HcclCcuUrmaPeerPlanAllToAllVExecuteLayout(void *send, void *recv,
    uint64_t sendCapacity, uint64_t recvCapacity, uint64_t rowBytes, uint32_t rank,
    HcclComm comm, aclrtStream stream, uint64_t handle, const uint64_t *counts, uint32_t count,
    const uint64_t *storageCounts, uint32_t storageCount)
{
    if (!storageCounts) return HCCL_E_PTR;
    return ExecuteAllToAllV(send, recv, sendCapacity, recvCapacity, rowBytes, rank,
        comm, stream, handle, counts, count, storageCounts, storageCount);
}

extern "C" HcclResult HcclCcuUrmaPeerPlanDestroy(uint64_t handle)
{
    std::lock_guard<std::mutex> build(buildMutex);
    std::lock_guard<std::mutex> submit(submitMutex);
    std::lock_guard<std::mutex> lock(registryMutex);
    const auto found = plans.find(handle);
    if (found == plans.end()) return HCCL_E_NOT_FOUND;
    for (auto &entry : found->second.bridges) {
        const auto status = entry.second->Close();
        if (status != HCCL_SUCCESS) return status;
    }
    for (auto &stream : found->second.streams) {
        if (stream.second.route.selfCopy) {
            const auto status = stream.second.route.selfCopy->Close();
            if (status != HCCL_SUCCESS) return status;
        }
    }
    plans.erase(found);
    return HCCL_SUCCESS;
}

extern "C" int A5CcuEidAllToAllVAbiVersion() { return 1; }
// Additive capability marker: no public task/layout ABI changes.
extern "C" int A5CcuEidV1CompatAbiVersion() { return 1; }
extern "C" int A5CcuEidPeerResourceAbiVersion() { return 1; }
extern "C" int A5CcuEidRegisterProbeAbiVersion() { return 1; }
extern "C" int A5CcuEidCompareProbeAbiVersion() { return 1; }
extern "C" uint32_t A5CcuEidDirectControlAbiVersion() { return 1; }

extern "C" HcclResult HcclCcuUrmaEidAllToAllVCreate(HcclComm comm, aclrtStream stream,
    const char* id, const char* manifest, uint64_t* handle) {
    return CreatePeerPlan(comm, stream, id, manifest, 8, handle, true, true);
}
extern "C" HcclResult HcclCcuUrmaEidAllToAllVExecute(void* send, void* recv,
    uint64_t sendCapacity, uint64_t recvCapacity, uint64_t rowBytes, uint32_t rank,
    HcclComm comm, aclrtStream stream, uint64_t handle, const uint64_t* counts,
    const int32_t* relays, const uint64_t* moved, uint32_t count) {
    if (!relays || !moved) return HCCL_E_PTR;
    return ExecuteAllToAllV(send, recv, sendCapacity, recvCapacity, rowBytes, rank,
                           comm, stream, handle, counts, count, nullptr, 0, relays, moved);
}

// Drain both caller and IO-die route streams before freeing payload sentinels.
extern "C" HcclResult HcclCcuUrmaPeerPlanSynchronize(uint64_t handle) {
    std::lock_guard<std::mutex> build(buildMutex);
    std::lock_guard<std::mutex> submit(submitMutex);
    std::vector<aclrtStream> streams;
    {
        std::lock_guard<std::mutex> lock(registryMutex);
        auto found=plans.find(handle);if(found==plans.end()) return HCCL_E_NOT_FOUND;
        int32_t device=-1;
        if(aclrtGetDevice(&device)!=ACL_SUCCESS || device!=found->second.device) return HCCL_E_PARA;
        for(const auto& entry:found->second.streams) {
            streams.push_back(reinterpret_cast<aclrtStream>(entry.first));
            if(entry.second.route.slaveStream) streams.push_back(entry.second.route.slaveStream);
        }
    }
    HcclResult status=HCCL_SUCCESS;
    for(auto stream:streams) if(aclrtSynchronizeStream(stream)!=ACL_SUCCESS) status=HCCL_E_RUNTIME;
    return status;
}
