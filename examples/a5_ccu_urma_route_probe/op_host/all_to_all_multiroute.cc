#include "a5_ccu_urma_route_probe.h"
#include "all_to_all_multiroute_kernel.h"
#include "utils.h"

#include <hcomm/ccu/hccl_ccu_res.h>
#include <hcomm/hcomm_res.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

using a5_ccu_urma_probe::AllToAllMultiRouteTaskArg;
using a5_ccu_urma_probe::GetRouteResources;
using a5_ccu_urma_probe::RouteKernelKind;
using a5_ccu_urma_probe::RoutePlanRequest;
using a5_ccu_urma_probe::RouteResources;

namespace {
constexpr uint32_t THREAD_NOTIFY_INDEX = 0;
constexpr uint32_t THREAD_NOTIFY_TIMEOUT = 1800;
constexpr uint64_t PATH_ALIGNMENT = 256;

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

HcclResult RunMultiRouteAllToAll(void *sendBuf, void *recvBuf,
    uint64_t elementsPerPeer, HcclDataType dataType, HcclComm comm,
    aclrtStream stream, const RoutePlanRequest *plan)
{
    if (sendBuf == nullptr || recvBuf == nullptr || comm == nullptr || stream == nullptr) {
        return HCCL_E_PTR;
    }
    if (dataType != HCCL_DATA_TYPE_FP32 || elementsPerPeer == 0) {
        return HCCL_E_NOT_SUPPORT;
    }

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
    if (status != HCCL_SUCCESS) {
        return status;
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
        pathWeights == nullptr || pathCount < 2U || pathCount > 8U) {
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
