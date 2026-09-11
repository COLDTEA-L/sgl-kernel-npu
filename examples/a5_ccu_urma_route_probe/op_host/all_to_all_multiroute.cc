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
}

extern "C" HcclResult HcclCcuUrmaMultiRouteAllToAll(void *sendBuf, void *recvBuf,
    uint64_t elementsPerPeer, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    if (sendBuf == nullptr || recvBuf == nullptr || comm == nullptr || stream == nullptr) {
        return HCCL_E_PTR;
    }
    if (dataType != HCCL_DATA_TYPE_FP32 || elementsPerPeer == 0) {
        return HCCL_E_NOT_SUPPORT;
    }

    RouteResources resources;
    HcclResult status = GetRouteResources(comm, stream, &resources);
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
        std::printf("[A5 CCU URMA A2A][rank=%u] launch kernel=%lu peer_bytes=%lu routes=%zu\n",
                    resources.rank, static_cast<unsigned long>(resources.allToAllKernel),
                    static_cast<unsigned long>(bytes), resources.channels.size());
    }
    status = HcclCcuKernelLaunch(comm, resources.routeThread,
                                resources.allToAllKernel, &taskArg);
    if (status != HCCL_SUCCESS || resources.routeThread == resources.mainThread) {
        return status;
    }
    status = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resources.mainThread, THREAD_NOTIFY_INDEX, THREAD_NOTIFY_TIMEOUT));
    if (status != HCCL_SUCCESS) return status;
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resources.routeThread, resources.mainThread, THREAD_NOTIFY_INDEX));
}
