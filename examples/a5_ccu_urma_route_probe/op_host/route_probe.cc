#include "a5_ccu_urma_route_probe.h"
#include "utils.h"

#include <hcomm/hcomm_primitives.h>

#include <cstdint>
#include <cstdio>

using a5_ccu_urma_probe::GetRouteResources;
using a5_ccu_urma_probe::RouteResources;

namespace {
constexpr uint32_t NOTIFY_INDEX = 0;
constexpr uint32_t NOTIFY_TIMEOUT_MS = 60000;

HcclResult ConvertHcommStatus(const char *operation, int32_t status)
{
    if (status == 0) {
        return HCCL_SUCCESS;
    }
    std::fprintf(stderr, "[A5 CCU URMA] %s failed, status=%d\n", operation, status);
    return HCCL_E_RUNTIME;
}
} // namespace

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
    HcclResult status = GetRouteResources(comm, stream, recvBuf, bytes * 2, &resources);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    if (resources.remoteRecv.type != COMM_MEM_TYPE_DEVICE || resources.remoteRecv.size < bytes * 2) {
        std::fprintf(stderr, "[A5 CCU URMA] invalid remote receive memory: type=%d size=%lu\n",
                     static_cast<int>(resources.remoteRecv.type),
                     static_cast<unsigned long>(resources.remoteRecv.size));
        return HCCL_E_PARA;
    }

    auto *localDst = static_cast<uint8_t *>(recvBuf) + resources.rank * bytes;
    auto *remoteDst = static_cast<uint8_t *>(resources.remoteRecv.addr) + resources.rank * bytes;
    status = ConvertHcommStatus("HcommLocalCopyOnThread",
        HcommLocalCopyOnThread(resources.thread, localDst, sendBuf, bytes));
    if (status != HCCL_SUCCESS) {
        return status;
    }
    status = ConvertHcommStatus("HcommWriteWithNotifyOnThread",
        HcommWriteWithNotifyOnThread(resources.thread, resources.channel, remoteDst,
                                     sendBuf, bytes, NOTIFY_INDEX));
    if (status != HCCL_SUCCESS) {
        return status;
    }
    return ConvertHcommStatus("HcommChannelNotifyWaitOnThread",
        HcommChannelNotifyWaitOnThread(resources.thread, resources.channel,
                                       NOTIFY_INDEX, NOTIFY_TIMEOUT_MS));
}
