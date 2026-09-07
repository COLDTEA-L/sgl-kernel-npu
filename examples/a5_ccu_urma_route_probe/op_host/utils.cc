#include "utils.h"
#include "route_kernel.h"

#include <hccl/hccl_rank_graph.h>
#include <hcomm/hcomm_res.h>
#include <hcomm/ccu/hccl_ccu_res.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace a5_ccu_urma_probe {
namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t EID_BYTE_NUM = 16;

struct ThreadLocalCache {
    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    uint32_t routeIndex = UINT32_MAX;
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
                         uint32_t routeIndex, ChannelHandle *channel)
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

    // HCCL's native A5 2Die collectives acquire every UBC_CTP link in the
    // selected network layer as one resource group.  A hop-2 layer can contain
    // one link per die; acquiring only one member times out even when the two
    // ranks chose symmetric endpoint pairs.  Acquire the whole layer, then
    // expose only the route requested by the user to the probe kernel.
    const uint32_t selectedLayer = candidateLayers[routeIndex];
    std::vector<HcclChannelDesc> descs;
    uint32_t selectedGroupIndex = UINT32_MAX;
    for (uint32_t ordinal = 0; ordinal < candidates.size(); ++ordinal) {
        if (candidateLayers[ordinal] != selectedLayer) {
            continue;
        }
        HcclChannelDesc desc;
        status = HcclChannelDescInit(&desc, 1);
        if (status != HCCL_SUCCESS) {
            return status;
        }
        const CommLink &link = candidates[ordinal];
        desc.remoteRank = peer;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint = link.srcEndpointDesc;
        desc.remoteEndpoint = link.dstEndpointDesc;
        if (ordinal == routeIndex) {
            selectedGroupIndex = static_cast<uint32_t>(descs.size());
        }
        descs.push_back(desc);
    }
    if (selectedGroupIndex == UINT32_MAX || descs.empty()) {
        return HCCL_E_INTERNAL;
    }

    std::vector<ChannelHandle> handles(descs.size(), 0);
    std::printf("[A5 CCU URMA][rank=%u peer=%u] acquiring route=%u layer=%u link=%u "
                "as layer group of %zu channel(s)\n",
                rank, peer, routeIndex, selectedLayer, candidateLinkIndices[routeIndex], descs.size());
    std::fflush(stdout);
    status = HcclChannelAcquire(comm, COMM_ENGINE_CCU, descs.data(),
                                static_cast<uint32_t>(descs.size()), handles.data());
    if (status == HCCL_SUCCESS) {
        *channel = handles[selectedGroupIndex];
    }
    std::printf("[A5 CCU URMA][rank=%u peer=%u] HcclChannelAcquire end: status=%d "
                "group_size=%zu selected_group_index=%u channel=%lu\n",
                rank, peer, static_cast<int>(status), handles.size(), selectedGroupIndex,
                static_cast<unsigned long>(status == HCCL_SUCCESS ? *channel : 0));
    std::fflush(stdout);
    return status;
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

HcclResult GetRouteResources(HcclComm comm, aclrtStream stream, RouteResources *resources)
{
    if (comm == nullptr || stream == nullptr || resources == nullptr) {
        return HCCL_E_PTR;
    }
    uint32_t routeIndex = 0;
    HcclResult status = GetCcuRouteIndex(&routeIndex);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    if (g_cache.comm == comm && g_cache.stream == stream && g_cache.routeIndex == routeIndex) {
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

    RouteResources created;
    created.rank = rank;
    created.rankSize = rankSize;
    std::printf("[A5 CCU URMA][rank=%u] HcclThreadAcquireWithStream(CCU) begin\n", rank);
    std::fflush(stdout);
    status = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream, 0, &created.thread);
    std::printf("[A5 CCU URMA][rank=%u] HcclThreadAcquireWithStream end: status=%d thread=%lu\n",
                rank, static_cast<int>(status), static_cast<unsigned long>(created.thread));
    std::fflush(stdout);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    status = SelectChannel(comm, rank, 1U - rank, routeIndex, &created.channel);
    if (status != HCCL_SUCCESS) {
        return status;
    }

    int32_t channelState = -1;
    const int32_t channelStatus = HcommChannelGetStatus(&created.channel, 1, &channelState);
    std::printf("[A5 CCU URMA][rank=%u] HcommChannelGetStatus: call_status=%d channel_state=%d\n",
                rank, channelStatus, channelState);
    std::fflush(stdout);
    if (channelStatus != 0) {
        return HCCL_E_RUNTIME;
    }

    RouteKernelArg kernelArg(created.channel, routeIndex);
    hcomm::KernelCreator creator = CreateRouteKernel;
    CcuKernelHandle kernel = 0;
    std::printf("[A5 CCU URMA][rank=%u] HcclCcuKernelRegister begin\n", rank);
    std::fflush(stdout);
    status = HcclCcuKernelRegister(comm, &kernel, &creator, &kernelArg);
    std::printf("[A5 CCU URMA][rank=%u] HcclCcuKernelRegister end: status=%d kernel=%lu\n",
                rank, static_cast<int>(status), static_cast<unsigned long>(kernel));
    std::fflush(stdout);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    created.kernel = kernel;

    std::printf("[A5 CCU URMA][rank=%u] HcclCcuKernelRegisterFinish begin\n", rank);
    std::fflush(stdout);
    status = HcclCcuKernelRegisterFinish(comm);
    std::printf("[A5 CCU URMA][rank=%u] HcclCcuKernelRegisterFinish end: status=%d\n",
                rank, static_cast<int>(status));
    std::fflush(stdout);
    if (status != HCCL_SUCCESS) {
        return status;
    }

    g_cache.comm = comm;
    g_cache.stream = stream;
    g_cache.routeIndex = routeIndex;
    g_cache.resources = created;
    *resources = created;
    return HCCL_SUCCESS;
}

} // namespace a5_ccu_urma_probe
