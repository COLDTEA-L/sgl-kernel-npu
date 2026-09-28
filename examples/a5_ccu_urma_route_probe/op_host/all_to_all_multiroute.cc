#include "a5_ccu_urma_route_probe.h"
#include "all_to_all_multiroute_kernel.h"
#include "command_block_worker_kernel.h"
#include "utils.h"

#include <hcomm/ccu/hccl_ccu_res.h>
#include <hcomm/hcomm_res.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using a5_ccu_urma_probe::AllToAllMultiRouteTaskArg;
using a5_ccu_urma_probe::CommandBlockWorkerTaskArg;
using a5_ccu_urma_probe::GetRouteResources;
using a5_ccu_urma_probe::RouteKernelKind;
using a5_ccu_urma_probe::RoutePlanRequest;
using a5_ccu_urma_probe::RouteResources;
using namespace a5_ccu_urma_probe;

namespace {
constexpr uint32_t MAX_EXPLICIT_PATHS = 64U;
constexpr uint32_t THREAD_NOTIFY_INDEX = 0;
constexpr uint32_t THREAD_NOTIFY_TIMEOUT = 1800;
constexpr uint64_t PATH_ALIGNMENT = 256;

struct PreparedPlan {
    HcclComm comm = nullptr;
    std::string planId;
    std::string fingerprint;
    std::string relayManifest;
    uint32_t directRoute = 0;
    std::vector<uint32_t> weights;
    std::unordered_map<uintptr_t, RouteResources> resourcesByStream;
};

struct CommandWorker {
    HcclComm comm = nullptr;
    aclrtStream workerStream = nullptr;
    RouteResources resources{};
    void *commandBlock = nullptr;
};

std::mutex g_planMutex;
std::mutex g_planBuildMutex;
std::unordered_map<uint64_t, PreparedPlan> g_plans;
std::unordered_map<std::string, uint64_t> g_planIds;
std::atomic<uint64_t> g_nextPlanHandle{1U};
std::unordered_map<uint64_t, CommandWorker> g_commandWorkers;
std::atomic<uint64_t> g_nextCommandWorkerHandle{1U};

bool DebugEnabled()
{
    const char *value = std::getenv("A5_CCU_DEBUG");
    return value != nullptr && std::string(value) == "1";
}

HcclResult GetSerializedSchedule(bool *serialized)
{
    if (serialized == nullptr) return HCCL_E_PTR;
    const char *value = std::getenv("A5_CCU_ROUTE_SCHEDULE");
    if (value == nullptr || value[0] == '\0' || std::string(value) == "concurrent") {
        *serialized = false;
        return HCCL_SUCCESS;
    }
    if (std::string(value) == "serial") {
        *serialized = true;
        return HCCL_SUCCESS;
    }
    std::fprintf(stderr, "[A5 CCU URMA] A5_CCU_ROUTE_SCHEDULE must be concurrent or serial\n");
    return HCCL_E_PARA;
}

std::string MakePlanIdentity(HcclComm comm, aclrtStream stream,
                             const std::string &planId)
{
    std::ostringstream output;
    output << reinterpret_cast<uintptr_t>(comm) << ':'
           << reinterpret_cast<uintptr_t>(stream) << ':' << planId;
    return output.str();
}

std::string MakePlanFingerprint(const char *relayManifest, uint32_t directRoute,
                                const uint32_t *pathWeights, uint32_t pathCount)
{
    std::ostringstream output;
    output << directRoute << ':' << relayManifest << ':';
    for (uint32_t i = 0; i < pathCount; ++i) output << pathWeights[i] << ',';
    return output.str();
}

HcclResult ValidatePlanArguments(HcclComm comm, aclrtStream stream,
                                 const char *planId, const char *relayManifest,
                                 const uint32_t *pathWeights, uint32_t pathCount)
{
    if (comm == nullptr || stream == nullptr || planId == nullptr ||
        relayManifest == nullptr || pathWeights == nullptr) {
        return HCCL_E_PTR;
    }
    if (planId[0] == '\0' || relayManifest[0] == '\0' ||
        pathCount < 2U || pathCount > MAX_EXPLICIT_PATHS) {
        return HCCL_E_PARA;
    }
    for (uint32_t i = 0; i < pathCount; ++i) {
        if (pathWeights[i] == 0U) return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchPreparedAllToAll(void *sendBuf, void *recvBuf,
    uint64_t elementsPerPeer, HcclDataType dataType, HcclComm comm,
    const RouteResources &resources, bool serialized)
{
    if (sendBuf == nullptr || recvBuf == nullptr || comm == nullptr) {
        return HCCL_E_PTR;
    }
    if (dataType != HCCL_DATA_TYPE_FP32 || elementsPerPeer == 0) {
        return HCCL_E_NOT_SUPPORT;
    }
    if (resources.channels.empty() || resources.weights.size() != resources.channels.size() ||
        resources.kernel == 0 || resources.rankSize != 2U) {
        return HCCL_E_INTERNAL;
    }

    const uint64_t bytes = elementsPerPeer * sizeof(float);
    const uint64_t totalBytes = bytes * resources.rankSize;
    const uint32_t peer = 1U - resources.rank;
    const uint64_t inputToken = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(sendBuf), totalBytes);
    const uint64_t outputToken = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(recvBuf), totalBytes);

    uint64_t totalWeight = 0;
    for (const uint32_t weight : resources.weights) {
        totalWeight += weight;
    }
    std::vector<uint64_t> sourceOffsets;
    std::vector<uint64_t> remoteOffsets;
    std::vector<uint64_t> pathBytes;
    uint64_t assigned = 0;
    for (size_t i = 0; i < resources.weights.size(); ++i) {
        uint64_t currentBytes = bytes - assigned;
        if (i + 1 != resources.weights.size()) {
            currentBytes = (bytes * resources.weights[i] / totalWeight) /
                           PATH_ALIGNMENT * PATH_ALIGNMENT;
        }
        sourceOffsets.push_back(static_cast<uint64_t>(peer) * bytes + assigned);
        remoteOffsets.push_back(static_cast<uint64_t>(resources.rank) * bytes + assigned);
        pathBytes.push_back(currentBytes);
        assigned += currentBytes;
    }

    AllToAllMultiRouteTaskArg taskArg(
        reinterpret_cast<uint64_t>(sendBuf), reinterpret_cast<uint64_t>(recvBuf),
        inputToken, outputToken, static_cast<uint64_t>(resources.rank) * bytes,
        static_cast<uint64_t>(resources.rank) * bytes, bytes,
        sourceOffsets, remoteOffsets, pathBytes);

    HcclResult status = HCCL_SUCCESS;
    if (resources.routeThread != resources.mainThread) {
        status = static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resources.mainThread, resources.routeThread, THREAD_NOTIFY_INDEX));
        if (status != HCCL_SUCCESS) return status;
        status = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resources.routeThread, THREAD_NOTIFY_INDEX, THREAD_NOTIFY_TIMEOUT));
        if (status != HCCL_SUCCESS) return status;
    }

    if (DebugEnabled()) {
        std::printf("[A5 CCU URMA A2A][rank=%u] launch kernel=%lu peer_bytes=%lu routes=%zu schedule=%s\n",
                    resources.rank,
                    static_cast<unsigned long>(resources.kernel),
                    static_cast<unsigned long>(bytes), resources.channels.size(),
                    serialized ? "serial" : "concurrent");
    }
    status = HcclCcuKernelLaunch(
        comm, resources.routeThread, resources.kernel, &taskArg);
    if (status != HCCL_SUCCESS || resources.routeThread == resources.mainThread) {
        return status;
    }
    status = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resources.mainThread, THREAD_NOTIFY_INDEX, THREAD_NOTIFY_TIMEOUT));
    if (status != HCCL_SUCCESS) return status;
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resources.routeThread, resources.mainThread, THREAD_NOTIFY_INDEX));
}

HcclResult RunMultiRouteAllToAll(void *sendBuf, void *recvBuf,
    uint64_t elementsPerPeer, HcclDataType dataType, HcclComm comm,
    aclrtStream stream, const RoutePlanRequest *plan)
{
    if (stream == nullptr) return HCCL_E_PTR;

    // A formal explicit plan is always concurrent: submit every remote write
    // before waiting for completion. The environment-selectable serial mode is
    // retained only by the legacy probe API as an experimental control.
    bool serialized = false;
    HcclResult status = HCCL_SUCCESS;
    if (plan == nullptr) {
        status = GetSerializedSchedule(&serialized);
        if (status != HCCL_SUCCESS) return status;
    }
    RouteResources resources;
    status = GetRouteResources(
        comm, stream,
        serialized ? RouteKernelKind::ALLTOALL_SERIAL
                   : RouteKernelKind::ALLTOALL_CONCURRENT,
        &resources, plan);
    if (status != HCCL_SUCCESS) return status;
    return LaunchPreparedAllToAll(sendBuf, recvBuf, elementsPerPeer, dataType,
                                  comm, resources, serialized);
}
}

extern "C" HcclResult HcclCcuUrmaMultiRouteAllToAll(void *sendBuf, void *recvBuf,
    uint64_t elementsPerPeer, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    return RunMultiRouteAllToAll(sendBuf, recvBuf, elementsPerPeer, dataType,
                                 comm, stream, nullptr);
}

extern "C" HcclResult HcclCcuUrmaExplicitMultipathAllToAll(
    void *sendBuf, void *recvBuf, uint64_t elementsPerPeer,
    HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const char *relayManifest, uint32_t directRoute,
    const uint32_t *pathWeights, uint32_t pathCount)
{
    if (relayManifest == nullptr || relayManifest[0] == '\0' ||
        pathWeights == nullptr || pathCount < 2U || pathCount > MAX_EXPLICIT_PATHS) {
        return HCCL_E_PARA;
    }
    RoutePlanRequest plan;
    plan.includeDiscoveredRoute = true;
    plan.discoveredRoute = directRoute;
    plan.relayManifest = relayManifest;
    plan.weights.assign(pathWeights, pathWeights + pathCount);
    plan.planName = "direct+explicit-relays";
    return RunMultiRouteAllToAll(sendBuf, recvBuf, elementsPerPeer, dataType,
                                 comm, stream, &plan);
}

extern "C" __attribute__((visibility("default"))) int A5CcuUrmaPreparedPlanAbiVersion()
{
    // Version 4 caches one prepared CCU resource set per execution stream so
    // torch.compile/npugraphs can warm up on its private graph stream before
    // capture without reusing a thread that was bound to the default stream.
    return 4;
}

extern "C" HcclResult HcclCcuUrmaExplicitMultipathPlanCreate(
    HcclComm comm, aclrtStream stream, const char *planId,
    const char *relayManifest, uint32_t directRoute,
    const uint32_t *pathWeights, uint32_t pathCount,
    uint64_t *planHandle)
{
    if (planHandle == nullptr) return HCCL_E_PTR;
    HcclResult status = ValidatePlanArguments(
        comm, stream, planId, relayManifest, pathWeights, pathCount);
    if (status != HCCL_SUCCESS) return status;

    const std::string identity = MakePlanIdentity(comm, stream, planId);
    const std::string fingerprint = MakePlanFingerprint(
        relayManifest, directRoute, pathWeights, pathCount);
    {
        std::lock_guard<std::mutex> lock(g_planMutex);
        const auto existing = g_planIds.find(identity);
        if (existing != g_planIds.end()) {
            const PreparedPlan &prepared = g_plans.at(existing->second);
            if (prepared.fingerprint != fingerprint) {
                std::fprintf(stderr,
                    "[A5 CCU URMA] prepared plan id %s is immutable; use a new id for a new path set\n",
                    planId);
                return HCCL_E_PARA;
            }
            *planHandle = existing->second;
            return HCCL_SUCCESS;
        }
    }

    RoutePlanRequest request;
    request.includeDiscoveredRoute = true;
    request.discoveredRoute = directRoute;
    request.relayManifest = relayManifest;
    request.weights.assign(pathWeights, pathWeights + pathCount);
    request.planName = planId;
    RouteResources resources;
    status = GetRouteResources(comm, stream, RouteKernelKind::ALLTOALL_CONCURRENT,
                               &resources, &request);
    if (status != HCCL_SUCCESS) return status;

    PreparedPlan prepared;
    prepared.comm = comm;
    prepared.planId = planId;
    prepared.fingerprint = fingerprint;
    prepared.relayManifest = relayManifest;
    prepared.directRoute = directRoute;
    prepared.weights.assign(pathWeights, pathWeights + pathCount);
    prepared.resourcesByStream.emplace(reinterpret_cast<uintptr_t>(stream), resources);
    uint64_t handle = g_nextPlanHandle.fetch_add(1U);
    if (handle == 0U) handle = g_nextPlanHandle.fetch_add(1U);
    {
        std::lock_guard<std::mutex> lock(g_planMutex);
        const auto existing = g_planIds.find(identity);
        if (existing != g_planIds.end()) {
            const PreparedPlan &installed = g_plans.at(existing->second);
            if (installed.fingerprint != fingerprint) return HCCL_E_PARA;
            *planHandle = existing->second;
            return HCCL_SUCCESS;
        }
        g_plans.emplace(handle, prepared);
        g_planIds.emplace(identity, handle);
    }
    *planHandle = handle;
    std::printf("PREPARED_MULTIPATH_PLAN plan_id=%s handle=%lu rank=%u paths=%zu die=%u\n",
                planId, static_cast<unsigned long>(handle), resources.rank,
                resources.channels.size(), resources.dieId);
    std::fflush(stdout);
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclCcuUrmaExplicitMultipathPlanExecute(
    void *sendBuf, void *recvBuf, uint64_t elementsPerPeer,
    HcclDataType dataType, HcclComm comm, aclrtStream stream,
    uint64_t planHandle)
{
    if (planHandle == 0U || stream == nullptr) return HCCL_E_PARA;
    const uintptr_t streamKey = reinterpret_cast<uintptr_t>(stream);
    HcclComm preparedComm = nullptr;
    RouteResources resources;
    bool foundResources = false;
    {
        std::lock_guard<std::mutex> lock(g_planMutex);
        const auto found = g_plans.find(planHandle);
        if (found == g_plans.end()) return HCCL_E_NOT_FOUND;
        const PreparedPlan &prepared = found->second;
        if (comm != nullptr && prepared.comm != comm) return HCCL_E_PARA;
        preparedComm = prepared.comm;
        const auto streamResources = prepared.resourcesByStream.find(streamKey);
        if (streamResources != prepared.resourcesByStream.end()) {
            resources = streamResources->second;
            foundResources = true;
        }
    }

    if (!foundResources) {
        // torch.compile/npugraphs performs an eager pre-execution on its graph
        // stream before capture. Build the stream-bound HCOMM resources once
        // in that pre-execution, then reuse the exact handles during capture
        // and replay. Serializing this slow path avoids leaking duplicate CCU
        // threads/channels if two host threads see the same stream at once.
        std::lock_guard<std::mutex> buildLock(g_planBuildMutex);
        RoutePlanRequest request;
        {
            std::lock_guard<std::mutex> lock(g_planMutex);
            const auto found = g_plans.find(planHandle);
            if (found == g_plans.end()) return HCCL_E_NOT_FOUND;
            const PreparedPlan &prepared = found->second;
            if (comm != nullptr && prepared.comm != comm) return HCCL_E_PARA;
            preparedComm = prepared.comm;
            const auto streamResources = prepared.resourcesByStream.find(streamKey);
            if (streamResources != prepared.resourcesByStream.end()) {
                resources = streamResources->second;
                foundResources = true;
            } else {
                request.includeDiscoveredRoute = true;
                request.discoveredRoute = prepared.directRoute;
                request.relayManifest = prepared.relayManifest;
                request.weights = prepared.weights;
                request.planName = prepared.planId;
            }
        }
        if (!foundResources) {
            std::printf("PREPARED_MULTIPATH_STREAM phase=begin handle=%lu stream=%p\n",
                        static_cast<unsigned long>(planHandle), stream);
            std::fflush(stdout);
            RouteResources created;
            const HcclResult status = GetRouteResources(
                preparedComm, stream, RouteKernelKind::ALLTOALL_CONCURRENT,
                &created, &request);
            if (status != HCCL_SUCCESS) return status;
            {
                std::lock_guard<std::mutex> lock(g_planMutex);
                const auto found = g_plans.find(planHandle);
                if (found == g_plans.end()) return HCCL_E_NOT_FOUND;
                found->second.resourcesByStream.emplace(streamKey, created);
            }
            resources = created;
            std::printf("PREPARED_MULTIPATH_STREAM phase=ready handle=%lu stream=%p rank=%u paths=%zu\n",
                        static_cast<unsigned long>(planHandle), stream,
                        resources.rank, resources.channels.size());
            std::fflush(stdout);
        }
    }
    return LaunchPreparedAllToAll(sendBuf, recvBuf, elementsPerPeer, dataType,
                                  preparedComm, resources, false);
}

extern "C" __attribute__((visibility("default"))) int
A5CcuHbmCommandPunctureAbiVersion()
{
    // Version 2 uses a single CCU polling loop.  The nested WHILE form in V1
    // was rejected by the A5 instruction builder during kernel registration.
    return 2;
}

extern "C" HcclResult HcclCcuUrmaCommandBlockWorkerCreate(
    void *sendBuf, void *recvBuf, uint64_t elementsPerPeer,
    HcclDataType dataType, void *commandBlock, uint64_t commandBlockBytes,
    HcclComm comm, aclrtStream controlStream, const char *planId,
    const char *relayManifest, uint32_t directRoute,
    const uint32_t *pathWeights, uint32_t pathCount,
    uint64_t *workerHandle)
{
    if (sendBuf == nullptr || recvBuf == nullptr || commandBlock == nullptr ||
        comm == nullptr || controlStream == nullptr || workerHandle == nullptr) {
        return HCCL_E_PTR;
    }
    if (dataType != HCCL_DATA_TYPE_FP32 || elementsPerPeer == 0 ||
        commandBlockBytes < COMMAND_WORDS * sizeof(uint64_t) ||
        planId == nullptr || planId[0] == '\0' || relayManifest == nullptr ||
        relayManifest[0] == '\0' || pathWeights == nullptr || pathCount < 2U ||
        pathCount > COMMAND_MAX_PATHS) {
        return HCCL_E_PARA;
    }
    for (uint32_t i = 0; i < pathCount; ++i) {
        if (pathWeights[i] == 0U) return HCCL_E_PARA;
    }

    aclrtStream workerStream = nullptr;
    if (aclrtCreateStream(&workerStream) != ACL_SUCCESS) return HCCL_E_RUNTIME;

    RoutePlanRequest request;
    request.includeDiscoveredRoute = true;
    request.discoveredRoute = directRoute;
    request.relayManifest = relayManifest;
    request.weights.assign(pathWeights, pathWeights + pathCount);
    request.planName = std::string(planId) + "-command-worker";
    request.commandBlockAddress = reinterpret_cast<uint64_t>(commandBlock);
    RouteResources resources;
    HcclResult status = GetRouteResources(
        comm, workerStream, RouteKernelKind::COMMAND_BLOCK_WORKER,
        &resources, &request);
    if (status != HCCL_SUCCESS) return status;

    std::vector<uint64_t> initial(COMMAND_WORDS, 0U);
    const uint64_t totalBytes = elementsPerPeer * resources.rankSize * sizeof(float);
    initial[COMMAND_MAGIC] = COMMAND_BLOCK_MAGIC;
    initial[COMMAND_INPUT_TOKEN] = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(sendBuf), totalBytes);
    initial[COMMAND_OUTPUT_TOKEN] = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(recvBuf), totalBytes);
    const aclError copyStatus = aclrtMemcpyAsync(
        commandBlock, commandBlockBytes, initial.data(),
        initial.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE,
        controlStream);
    if (copyStatus != ACL_SUCCESS || aclrtSynchronizeStream(controlStream) != ACL_SUCCESS) {
        return HCCL_E_RUNTIME;
    }

    if (resources.routeThread != resources.mainThread) {
        status = static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resources.mainThread, resources.routeThread,
            THREAD_NOTIFY_INDEX));
        if (status != HCCL_SUCCESS) return status;
        status = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resources.routeThread, THREAD_NOTIFY_INDEX,
            THREAD_NOTIFY_TIMEOUT));
        if (status != HCCL_SUCCESS) return status;
    }

    CommandBlockWorkerTaskArg taskArg(reinterpret_cast<uint64_t>(commandBlock));
    status = HcclCcuKernelLaunch(
        comm, resources.routeThread, resources.kernel, &taskArg);
    if (status != HCCL_SUCCESS) return status;

    uint64_t handle = g_nextCommandWorkerHandle.fetch_add(1U);
    if (handle == 0U) handle = g_nextCommandWorkerHandle.fetch_add(1U);
    CommandWorker worker;
    worker.comm = comm;
    worker.workerStream = workerStream;
    worker.resources = resources;
    worker.commandBlock = commandBlock;
    {
        std::lock_guard<std::mutex> lock(g_planMutex);
        g_commandWorkers.emplace(handle, worker);
    }
    *workerHandle = handle;
    std::printf("COMMAND_BLOCK_WORKER phase=ready handle=%lu rank=%u paths=%zu "
                "command_block=%p stream=%p\n",
                static_cast<unsigned long>(handle), resources.rank,
                resources.channels.size(), commandBlock, workerStream);
    std::fflush(stdout);
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclCcuUrmaCommandBlockWorkerStop(
    uint64_t workerHandle, aclrtStream controlStream)
{
    if (workerHandle == 0U || controlStream == nullptr) return HCCL_E_PARA;
    CommandWorker worker;
    {
        std::lock_guard<std::mutex> lock(g_planMutex);
        const auto found = g_commandWorkers.find(workerHandle);
        if (found == g_commandWorkers.end()) return HCCL_E_NOT_FOUND;
        worker = found->second;
    }
    const uint64_t zero = 0;
    const uint64_t stop = COMMAND_OPCODE_STOP;
    auto *base = reinterpret_cast<unsigned char *>(worker.commandBlock);
    aclError aclStatus = aclrtMemcpyAsync(
        base + COMMAND_COMPLETION * sizeof(uint64_t), sizeof(uint64_t),
        &zero, sizeof(zero), ACL_MEMCPY_HOST_TO_DEVICE, controlStream);
    if (aclStatus == ACL_SUCCESS) {
        aclStatus = aclrtMemcpyAsync(
            base + COMMAND_OPCODE * sizeof(uint64_t), sizeof(uint64_t),
            &stop, sizeof(stop), ACL_MEMCPY_HOST_TO_DEVICE, controlStream);
    }
    if (aclStatus != ACL_SUCCESS || aclrtSynchronizeStream(controlStream) != ACL_SUCCESS) {
        return HCCL_E_RUNTIME;
    }
    aclrtStream executionStream = worker.resources.slaveStream != nullptr ?
        worker.resources.slaveStream : worker.workerStream;
    if (aclrtSynchronizeStream(executionStream) != ACL_SUCCESS) return HCCL_E_RUNTIME;
    {
        std::lock_guard<std::mutex> lock(g_planMutex);
        g_commandWorkers.erase(workerHandle);
    }
    std::printf("COMMAND_BLOCK_WORKER phase=stopped handle=%lu\n",
                static_cast<unsigned long>(workerHandle));
    std::fflush(stdout);
    return HCCL_SUCCESS;
}
