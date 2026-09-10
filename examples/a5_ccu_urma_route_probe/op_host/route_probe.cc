#include "a5_ccu_urma_route_probe.h"
#include "route_kernel.h"
#include "utils.h"

#include <hcomm/ccu/hccl_ccu_res.h>
#include <hcomm/hcomm_res.h>

#include <cstdint>
#include <cstdio>

using a5_ccu_urma_probe::GetRouteResources;
using a5_ccu_urma_probe::RouteResources;

namespace {
constexpr uint32_t THREAD_NOTIFY_INDEX = 0;
constexpr uint32_t THREAD_NOTIFY_TIMEOUT = 1800;
}

extern "C" HcclResult HcclCcuUrmaRouteProbe(void *sendBuf, void *recvBuf,
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
    HcclResult status = GetRouteResources(comm, stream, &resources);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    auto *localDst = static_cast<uint8_t *>(recvBuf) + resources.rank * bytes;
    const aclError copyStatus = aclrtMemcpyAsync(localDst, bytes, sendBuf, bytes,
                                                  ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    if (copyStatus != ACL_SUCCESS) {
        std::fprintf(stderr, "[A5 CCU URMA][rank=%u] self copy failed: status=%d\n",
                     resources.rank, static_cast<int>(copyStatus));
        return HCCL_E_RUNTIME;
    }

    const uint64_t inputToken = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(sendBuf), bytes);
    const uint64_t outputToken = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(recvBuf), bytes * resources.rankSize);
    a5_ccu_urma_probe::RouteTaskArg taskArg(
        reinterpret_cast<uint64_t>(sendBuf), reinterpret_cast<uint64_t>(recvBuf),
        inputToken, outputToken, bytes, resources.rank * bytes);

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

    std::printf("[A5 CCU URMA][rank=%u] HcclCcuKernelLaunch begin: kernel=%lu bytes=%lu die_id=%u\n",
                resources.rank, static_cast<unsigned long>(resources.kernel),
                static_cast<unsigned long>(bytes), resources.dieId);
    std::fflush(stdout);
    status = HcclCcuKernelLaunch(comm, resources.routeThread, resources.kernel, &taskArg);
    std::printf("[A5 CCU URMA][rank=%u] HcclCcuKernelLaunch end: status=%d\n",
                resources.rank, static_cast<int>(status));
    std::fflush(stdout);
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
