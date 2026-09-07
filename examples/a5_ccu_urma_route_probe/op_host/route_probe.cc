#include "a5_ccu_urma_route_probe.h"
#include "utils.h"

#include <hcomm/hcomm_primitives.h>

#include <cstdint>
#include <cstdio>

using a5_ccu_urma_probe::GetRouteResources;
using a5_ccu_urma_probe::RouteResources;

namespace {
constexpr uint64_t CCL_BUFFER_ALIGNMENT = 512;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

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
    HcclResult status = GetRouteResources(comm, stream, &resources);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    const uint64_t recvOffset = AlignUp(bytes, CCL_BUFFER_ALIGNMENT);
    const uint64_t requiredBytes = recvOffset + bytes;
    if (resources.localCclBufferSize < requiredBytes || resources.remoteCclBufferSize < requiredBytes) {
        std::fprintf(stderr,
            "[A5 CCU URMA] CCL buffer too small: required=%lu local=%lu remote=%lu\n",
            static_cast<unsigned long>(requiredBytes),
            static_cast<unsigned long>(resources.localCclBufferSize),
            static_cast<unsigned long>(resources.remoteCclBufferSize));
        return HCCL_E_PARA;
    }

    auto *localSend = static_cast<uint8_t *>(resources.localCclBuffer);
    auto *remoteRecv = static_cast<uint8_t *>(resources.remoteCclBuffer) + recvOffset;
    auto *localDst = static_cast<uint8_t *>(recvBuf) + resources.rank * bytes;
    status = ConvertHcommStatus("HcommLocalCopyOnThread",
        HcommLocalCopyOnThread(resources.thread, localSend, sendBuf, bytes));
    if (status != HCCL_SUCCESS) {
        return status;
    }
    status = ConvertHcommStatus("HcommLocalCopyOnThread(self)",
        HcommLocalCopyOnThread(resources.thread, localDst, sendBuf, bytes));
    if (status != HCCL_SUCCESS) {
        return status;
    }
    std::printf("[A5 CCU URMA][rank=%u] HcommWriteOnThread begin: bytes=%lu\n",
                resources.rank, static_cast<unsigned long>(bytes));
    std::fflush(stdout);
    status = ConvertHcommStatus("HcommWriteOnThread",
        HcommWriteOnThread(resources.thread, resources.channel, remoteRecv, localSend, bytes));
    std::printf("[A5 CCU URMA][rank=%u] HcommWriteOnThread end: status=%d\n",
                resources.rank, static_cast<int>(status));
    std::fflush(stdout);
    return status;
}

extern "C" HcclResult HcclCcuUrmaRouteProbeReadback(void *recvBuf,
    uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    if (recvBuf == nullptr || comm == nullptr || stream == nullptr) {
        return HCCL_E_PTR;
    }
    if (dataType != HCCL_DATA_TYPE_FP32) {
        return HCCL_E_NOT_SUPPORT;
    }

    const uint64_t bytes = sendCount * sizeof(float);
    RouteResources resources;
    HcclResult status = GetRouteResources(comm, stream, &resources);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    const uint64_t recvOffset = AlignUp(bytes, CCL_BUFFER_ALIGNMENT);
    if (resources.localCclBufferSize < recvOffset + bytes) {
        return HCCL_E_PARA;
    }
    const uint32_t peer = 1U - resources.rank;
    auto *localRecv = static_cast<uint8_t *>(resources.localCclBuffer) + recvOffset;
    auto *output = static_cast<uint8_t *>(recvBuf) + peer * bytes;
    return ConvertHcommStatus("HcommLocalCopyOnThread(readback)",
        HcommLocalCopyOnThread(resources.thread, output, localRecv, bytes));
}
