#include "utils.h"

#include <hccl/hccl_rank_graph.h>
#include <hcomm/hcomm_primitives.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace a5_ccu_urma_probe {
namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t EID_BYTE_NUM = 16;
constexpr char RECV_MEM_TAG[] = "a5_ccu_urma_route_probe_recv";

struct ThreadLocalCache {
    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    void *recvBuf = nullptr;
    uint64_t recvBytes = 0;
    uint32_t routeIndex = UINT32_MAX;
    HcclMemHandle memHandle = nullptr;
    RouteResources resources{};
};

thread_local ThreadLocalCache g_cache;

std::string CommAddrToString(const CommAddr &addr)
{
    std::ostringstream out;
    out << "type=" << static_cast<int>(addr.type) << ",raw="
        << std::hex << std::setfill('0');
    for (uint32_t i = 0; i < EID_BYTE_NUM; ++i) {
        if (i != 0 && (i % 2) == 0) {
            out << ':';
        }
        out << std::setw(2) << static_cast<uint32_t>(addr.eid[i]);
    }
    return out.str();
}

HcclResult SelectChannel(HcclComm comm, uint32_t rank, uint32_t peer,
                         uint32_t routeIndex, HcclMemHandle memHandle,
                         ChannelHandle *channel)
{
    uint32_t layerNum = 0;
    uint32_t *layers = nullptr;
    HcclResult status = HcclRankGraphGetLayers(comm, &layers, &layerNum);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    if (layerNum == 0 || layers == nullptr) {
        std::fprintf(stderr, "[A5 CCU URMA] rank graph contains no network layer\n");
        return HCCL_E_NOT_FOUND;
    }

    std::vector<CommLink> candidates;
    std::vector<uint32_t> candidateLayers;
    std::vector<uint32_t> candidateLinkIndices;
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        uint32_t listSize = 0;
        CommLink *linkList = nullptr;
        status = HcclRankGraphGetLinks(comm, layers[layerIndex], rank, peer, &linkList, &listSize);
        if (status != HCCL_SUCCESS) {
            return status;
        }
        for (uint32_t linkIndex = 0; linkIndex < listSize; ++linkIndex) {
            const CommLink &link = linkList[linkIndex];
            if (link.linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
            const uint32_t ordinal = static_cast<uint32_t>(candidates.size());
            const std::string srcAddr = CommAddrToString(link.srcEndpointDesc.commAddr);
            const std::string dstAddr = CommAddrToString(link.dstEndpointDesc.commAddr);
            std::printf("[A5 CCU URMA][rank=%u peer=%u] route=%u layer=%u link=%u "
                        "protocol=%d hop=%u src_phy=%u dst_phy=%u src_addr=%s dst_addr=%s%s\n",
                        rank, peer, ordinal, layers[layerIndex], linkIndex,
                        static_cast<int>(link.linkAttr.linkProtocol),
                        static_cast<uint32_t>(link.linkAttr.hop),
                        link.srcEndpointDesc.loc.device.devPhyId,
                        link.dstEndpointDesc.loc.device.devPhyId,
                        srcAddr.c_str(), dstAddr.c_str(),
                        ordinal == routeIndex ? " SELECTED" : "");
            std::fflush(stdout);
            candidates.push_back(link);
            candidateLayers.push_back(layers[layerIndex]);
            candidateLinkIndices.push_back(linkIndex);
        }
    }

    if (routeIndex >= candidates.size()) {
        std::fprintf(stderr,
            "[A5 CCU URMA] route index %u is out of range; rank %u -> %u has %zu UBC_CTP route(s)\n",
            routeIndex, rank, peer, candidates.size());
        return candidates.empty() ? HCCL_E_NOT_FOUND : HCCL_E_PARA;
    }

    const CommLink &link = candidates[routeIndex];
    HcclChannelDesc desc;
    status = HcclChannelDescInit(&desc, 1);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    desc.remoteRank = peer;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.memHandles = &memHandle;
    desc.memHandleNum = 1;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint = link.srcEndpointDesc;
    desc.remoteEndpoint = link.dstEndpointDesc;
    std::printf("[A5 CCU URMA][rank=%u peer=%u] acquiring route=%u layer=%u link=%u\n",
                rank, peer, routeIndex, candidateLayers[routeIndex], candidateLinkIndices[routeIndex]);
    std::fflush(stdout);
    return HcclChannelAcquire(comm, COMM_ENGINE_CCU, &desc, 1, channel);
}

HcclResult FindRemoteRecv(HcclComm comm, ChannelHandle channel, CommMem *remoteRecv)
{
    uint32_t memNum = 0;
    CommMem *remoteMems = nullptr;
    char **memTags = nullptr;
    HcclResult status = HcclChannelGetRemoteMems(comm, channel, &memNum, &remoteMems, &memTags);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    for (uint32_t i = 0; i < memNum; ++i) {
        if (memTags[i] != nullptr && std::strcmp(memTags[i], RECV_MEM_TAG) == 0) {
            *remoteRecv = remoteMems[i];
            return HCCL_SUCCESS;
        }
    }
    std::fprintf(stderr, "[A5 CCU URMA] remote memory tag %s not found; memNum=%u\n",
                 RECV_MEM_TAG, memNum);
    return HCCL_E_NOT_FOUND;
}

} // namespace

HcclResult GetCcuRouteIndex(uint32_t *routeIndex)
{
    if (routeIndex == nullptr) {
        return HCCL_E_PTR;
    }
    const char *value = std::getenv("A5_CCU_ROUTE_INDEX");
    if (value == nullptr || value[0] == '\0') {
        *routeIndex = 0;
        return HCCL_SUCCESS;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > UINT32_MAX) {
        std::fprintf(stderr, "[A5 CCU URMA] invalid A5_CCU_ROUTE_INDEX=%s\n", value);
        return HCCL_E_PARA;
    }
    *routeIndex = static_cast<uint32_t>(parsed);
    return HCCL_SUCCESS;
}

HcclResult GetRouteResources(HcclComm comm, aclrtStream stream, void *recvBuf,
                             uint64_t recvBytes, RouteResources *resources)
{
    if (comm == nullptr || stream == nullptr || recvBuf == nullptr || resources == nullptr) {
        return HCCL_E_PTR;
    }
    uint32_t routeIndex = 0;
    HcclResult status = GetCcuRouteIndex(&routeIndex);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    if (g_cache.comm == comm && g_cache.stream == stream && g_cache.recvBuf == recvBuf &&
        g_cache.recvBytes == recvBytes && g_cache.routeIndex == routeIndex) {
        *resources = g_cache.resources;
        return HCCL_SUCCESS;
    }

    uint32_t rank = 0;
    uint32_t rankSize = 0;
    status = HcclGetRankId(comm, &rank);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    status = HcclGetRankSize(comm, &rankSize);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    if (rankSize != 2) {
        std::fprintf(stderr, "[A5 CCU URMA] route probe requires rankSize=2; got %u\n", rankSize);
        return HCCL_E_PARA;
    }

    CommMem localRecv{COMM_MEM_TYPE_DEVICE, recvBuf, recvBytes};
    HcclMemHandle memHandle = nullptr;
    status = HcclCommMemReg(comm, RECV_MEM_TAG, &localRecv, &memHandle);
    if (status != HCCL_SUCCESS) {
        return status;
    }

    RouteResources created;
    created.rank = rank;
    created.rankSize = rankSize;
    status = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream, 0, &created.thread);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    status = SelectChannel(comm, rank, 1U - rank, routeIndex, memHandle, &created.channel);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    status = FindRemoteRecv(comm, created.channel, &created.remoteRecv);
    if (status != HCCL_SUCCESS) {
        return status;
    }

    g_cache.comm = comm;
    g_cache.stream = stream;
    g_cache.recvBuf = recvBuf;
    g_cache.recvBytes = recvBytes;
    g_cache.routeIndex = routeIndex;
    g_cache.memHandle = memHandle;
    g_cache.resources = created;
    *resources = created;
    return HCCL_SUCCESS;
}

} // namespace a5_ccu_urma_probe
