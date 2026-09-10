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
constexpr uint32_t THREAD_NOTIFY_NUM = 1;
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

HcclResult SelectRoute(HcclComm comm, uint32_t rank, uint32_t peer,
                         uint32_t routeIndex, HcclChannelDesc *desc,
                         uint32_t *dieId)
{
    if (desc == nullptr || dieId == nullptr) {
        return HCCL_E_PTR;
    }
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
            uint32_t srcDieId = UINT32_MAX;
            uint32_t dstDieId = UINT32_MAX;
            EndpointDesc srcEndpoint = link.srcEndpointDesc;
            EndpointDesc dstEndpoint = link.dstEndpointDesc;
            (void)HcclRankGraphGetEndpointInfo(
                comm, rank, &srcEndpoint, ENDPOINT_ATTR_DIE_ID,
                sizeof(srcDieId), static_cast<void *>(&srcDieId));
            (void)HcclRankGraphGetEndpointInfo(
                comm, peer, &dstEndpoint, ENDPOINT_ATTR_DIE_ID,
                sizeof(dstDieId), static_cast<void *>(&dstDieId));
            std::printf("[A5 CCU URMA][rank=%u peer=%u] route=%u layer=%u link=%u "
                        "protocol=%d hop=%u src_phy=%u dst_phy=%u src_die=%u dst_die=%u "
                        "src_addr=%s dst_addr=%s%s\n",
                        rank, peer, ordinal, layers[layerIndex], linkIndex,
                        static_cast<int>(link.linkAttr.linkProtocol),
                        static_cast<uint32_t>(link.linkAttr.hop),
                        link.srcEndpointDesc.loc.device.devPhyId,
                        link.dstEndpointDesc.loc.device.devPhyId,
                        srcDieId, dstDieId,
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

    const uint32_t selectedLayer = candidateLayers[routeIndex];
    const CommLink &selectedLink = candidates[routeIndex];
    EndpointDesc selectedEndpoint = selectedLink.srcEndpointDesc;
    status = HcclRankGraphGetEndpointInfo(
        comm, rank, &selectedEndpoint, ENDPOINT_ATTR_DIE_ID,
        sizeof(*dieId), static_cast<void *>(dieId));
    if (status != HCCL_SUCCESS) {
        std::fprintf(stderr,
            "[A5 CCU URMA][rank=%u peer=%u] get selected endpoint die id failed: status=%d\n",
            rank, peer, static_cast<int>(status));
        return status;
    }
    if (*dieId > 1) {
        std::fprintf(stderr,
            "[A5 CCU URMA][rank=%u peer=%u] unsupported endpoint die id %u\n",
            rank, peer, *dieId);
        return HCCL_E_PARA;
    }

    status = HcclChannelDescInit(desc, 1);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    desc->remoteRank = peer;
    desc->notifyNum = CHANNEL_NOTIFY_NUM;
    desc->channelProtocol = selectedLink.linkAttr.linkProtocol;
    desc->localEndpoint = selectedLink.srcEndpointDesc;
    desc->remoteEndpoint = selectedLink.dstEndpointDesc;

    std::printf("[A5 CCU URMA][rank=%u peer=%u] prepared route=%u layer=%u link=%u "
                "die_id=%u; threads will be acquired before this per-die channel\n",
                rank, peer, routeIndex, selectedLayer, candidateLinkIndices[routeIndex], *dieId);
    std::fflush(stdout);
    return HCCL_SUCCESS;
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
    HcclChannelDesc selectedDesc;
    status = SelectRoute(comm, rank, 1U - rank, routeIndex,
                         &selectedDesc, &created.dieId);
    if (status != HCCL_SUCCESS) {
        return status;
    }

    std::printf("[A5 CCU URMA][rank=%u] acquire main CCU thread begin\n", rank);
    std::fflush(stdout);
    status = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream,
                                        THREAD_NOTIFY_NUM, &created.mainThread);
    std::printf("[A5 CCU URMA][rank=%u] acquire main CCU thread end: status=%d thread=%lu\n",
                rank, static_cast<int>(status), static_cast<unsigned long>(created.mainThread));
    std::fflush(stdout);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    created.routeThread = created.mainThread;
    if (created.dieId == 1) {
        const aclError aclStatus = aclrtCreateStream(&created.slaveStream);
        if (aclStatus != ACL_SUCCESS) {
            std::fprintf(stderr, "[A5 CCU URMA][rank=%u] create die1 slave stream failed: status=%d\n",
                         rank, static_cast<int>(aclStatus));
            return HCCL_E_RUNTIME;
        }
        std::printf("[A5 CCU URMA][rank=%u] acquire die1 slave CCU thread begin\n", rank);
        std::fflush(stdout);
        status = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, created.slaveStream,
                                            THREAD_NOTIFY_NUM, &created.routeThread);
        std::printf("[A5 CCU URMA][rank=%u] acquire die1 slave CCU thread end: status=%d thread=%lu\n",
                    rank, static_cast<int>(status),
                    static_cast<unsigned long>(created.routeThread));
        std::fflush(stdout);
        if (status != HCCL_SUCCESS) {
            return status;
        }
    }

    std::printf("[A5 CCU URMA][rank=%u peer=%u] acquiring route=%u die_id=%u "
                "after all required CCU threads are ready\n",
                rank, 1U - rank, routeIndex, created.dieId);
    std::fflush(stdout);
    status = HcclChannelAcquire(comm, COMM_ENGINE_CCU, &selectedDesc, 1, &created.channel);
    std::printf("[A5 CCU URMA][rank=%u peer=%u] HcclChannelAcquire end: status=%d "
                "die_id=%u channel=%lu\n",
                rank, 1U - rank, static_cast<int>(status), created.dieId,
                static_cast<unsigned long>(status == HCCL_SUCCESS ? created.channel : 0));
    std::fflush(stdout);
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
