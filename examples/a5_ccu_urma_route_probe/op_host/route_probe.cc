#include "a5_ccu_urma_route_probe.h"
#include "route_kernel.h"
#include "one_way_write_kernel.h"
#include "utils.h"

#include <hcomm/ccu/hccl_ccu_res.h>
#include <hcomm/hcomm_res.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using a5_ccu_urma_probe::GetRouteResources;
using a5_ccu_urma_probe::RouteKernelKind;
using a5_ccu_urma_probe::RouteResources;

namespace {
constexpr uint32_t THREAD_NOTIFY_INDEX = 0;
constexpr uint32_t THREAD_NOTIFY_TIMEOUT = 1800;
constexpr uint64_t PATH_ALIGNMENT = 256;

bool EnvEnabled(const char *name)
{
    const char *value = std::getenv(name);
    return value != nullptr && std::string(value) == "1";
}

bool DebugEnabled()
{
    return EnvEnabled("A5_CCU_DEBUG");
}
}

extern "C" HcclResult HcclCcuUrmaMultiRouteWrite(void *sendBuf, void *recvBuf,
    uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    if (sendBuf == nullptr || recvBuf == nullptr || comm == nullptr || stream == nullptr) {
        return HCCL_E_PTR;
    }
    if (dataType != HCCL_DATA_TYPE_FP32) {
        std::fprintf(stderr, "[A5 CCU URMA] probe currently supports FP32 only\n");
        return HCCL_E_NOT_SUPPORT;
    }

    const uint64_t bytes = sendCount * sizeof(float);
    RouteResources resources;
    HcclResult status = GetRouteResources(
        comm, stream, RouteKernelKind::ROUTE_WRITE, &resources);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    if (EnvEnabled("A5_CCU_CHANNEL_ONLY")) {
        return HCCL_SUCCESS;
    }
    auto *localDst = static_cast<uint8_t *>(recvBuf) + resources.rank * bytes;
    if (!EnvEnabled("A5_CCU_REMOTE_ONLY")) {
        const aclError copyStatus = aclrtMemcpyAsync(localDst, bytes, sendBuf, bytes,
                                                      ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        if (copyStatus != ACL_SUCCESS) {
            std::fprintf(stderr, "[A5 CCU URMA][rank=%u] self copy failed: status=%d\n",
                         resources.rank, static_cast<int>(copyStatus));
            return HCCL_E_RUNTIME;
        }
    }

    const uint64_t inputToken = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(sendBuf), bytes);
    const uint64_t outputToken = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(recvBuf), bytes * resources.rankSize);
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
            currentBytes = (bytes * resources.weights[i] / totalWeight) / PATH_ALIGNMENT * PATH_ALIGNMENT;
        }
        sourceOffsets.push_back(assigned);
        remoteOffsets.push_back(resources.rank * bytes + assigned);
        pathBytes.push_back(currentBytes);
        assigned += currentBytes;
    }
    a5_ccu_urma_probe::RouteTaskArg taskArg(
        reinterpret_cast<uint64_t>(sendBuf), reinterpret_cast<uint64_t>(recvBuf),
        inputToken, outputToken, sourceOffsets, remoteOffsets, pathBytes);

    if (resources.routeThread != resources.mainThread) {
        status = static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resources.mainThread, resources.routeThread, THREAD_NOTIFY_INDEX));
        if (status != HCCL_SUCCESS) {
            return status;
        }
        status = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resources.routeThread, THREAD_NOTIFY_INDEX, THREAD_NOTIFY_TIMEOUT));
        if (status != HCCL_SUCCESS) {
            return status;
        }
    }

    if (DebugEnabled()) {
        std::printf("[A5 CCU URMA][rank=%u] HcclCcuKernelLaunch begin: kernel=%lu bytes=%lu die_id=%u\n",
                    resources.rank, static_cast<unsigned long>(resources.kernel),
                    static_cast<unsigned long>(bytes), resources.dieId);
    }
    status = HcclCcuKernelLaunch(comm, resources.routeThread, resources.kernel, &taskArg);
    if (DebugEnabled()) {
        std::printf("[A5 CCU URMA][rank=%u] HcclCcuKernelLaunch end: status=%d\n",
                    resources.rank, static_cast<int>(status));
    }
    if (status != HCCL_SUCCESS || resources.routeThread == resources.mainThread) {
        return status;
    }

    status = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resources.mainThread, THREAD_NOTIFY_INDEX, THREAD_NOTIFY_TIMEOUT));
    if (status != HCCL_SUCCESS) {
        return status;
    }
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resources.routeThread, resources.mainThread, THREAD_NOTIFY_INDEX));
}

extern "C" HcclResult HcclCcuUrmaRouteProbe(void *sendBuf, void *recvBuf,
    uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    return HcclCcuUrmaMultiRouteWrite(sendBuf, recvBuf, sendCount, dataType, comm, stream);
}

extern "C" HcclResult HcclCcuUrmaOneWayWrite(void *sendBuf, void *recvBuf,
    uint64_t sendCount, HcclDataType dataType, uint32_t sourceRank,
    HcclComm comm, aclrtStream stream)
{
    if (sendBuf == nullptr || recvBuf == nullptr || comm == nullptr || stream == nullptr)
        return HCCL_E_PTR;
    if (dataType != HCCL_DATA_TYPE_FP32) return HCCL_E_NOT_SUPPORT;
    RouteResources resources;
    HcclResult status = GetRouteResources(comm, stream, RouteKernelKind::ONE_WAY_WRITE,
                                          &resources, static_cast<int32_t>(sourceRank));
    if (status != HCCL_SUCCESS) return status;
    if (sourceRank >= resources.rankSize || resources.channels.size() != 1) return HCCL_E_PARA;
    const uint64_t bytes = sendCount * sizeof(float);
    const uint64_t inputToken = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(sendBuf), bytes);
    const uint64_t outputToken = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(recvBuf), bytes);
    a5_ccu_urma_probe::OneWayWriteTaskArg taskArg(
        reinterpret_cast<uint64_t>(sendBuf), reinterpret_cast<uint64_t>(recvBuf),
        inputToken, outputToken, bytes);
    if (resources.routeThread != resources.mainThread) {
        status = static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resources.mainThread, resources.routeThread, THREAD_NOTIFY_INDEX));
        if (status != HCCL_SUCCESS) return status;
        status = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resources.routeThread, THREAD_NOTIFY_INDEX, THREAD_NOTIFY_TIMEOUT));
        if (status != HCCL_SUCCESS) return status;
    }
    status = HcclCcuKernelLaunch(comm, resources.routeThread, resources.kernel, &taskArg);
    if (status != HCCL_SUCCESS || resources.routeThread == resources.mainThread) return status;
    status = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resources.mainThread, THREAD_NOTIFY_INDEX, THREAD_NOTIFY_TIMEOUT));
    if (status != HCCL_SUCCESS) return status;
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resources.routeThread, resources.mainThread, THREAD_NOTIFY_INDEX));
}
