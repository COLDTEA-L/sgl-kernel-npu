#include "utils.h"
#include "route_kernel.h"
#include "source_route_provider.h"

#include <hccl/hccl_rank_graph.h>
#include <hcomm/hcomm_res.h>
#include <hcomm/ccu/hccl_ccu_res.h>

#include <cerrno>
#include <algorithm>
#include <chrono>
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
    std::string routeKey;
    RouteResources resources{};
};

thread_local ThreadLocalCache g_cache;

bool EnvEnabled(const char *name)
{
    const char *value = std::getenv(name);
    return value != nullptr && std::string(value) == "1";
}

struct EndpointInfo {
    uint32_t dieId = UINT32_MAX;
    uint32_t bwCoeff = UINT32_MAX;
    uint32_t location = UINT32_MAX;
    HcclResult dieStatus = HCCL_E_NOT_FOUND;
    HcclResult bwStatus = HCCL_E_NOT_FOUND;
    HcclResult locationStatus = HCCL_E_NOT_FOUND;
};

EndpointInfo QueryEndpointInfo(HcclComm comm, uint32_t ownerRank,
                               const EndpointDesc &endpoint)
{
    EndpointInfo info;
    EndpointDesc endpointCopy = endpoint;
    info.dieStatus = HcclRankGraphGetEndpointInfo(
        comm, ownerRank, &endpointCopy, ENDPOINT_ATTR_DIE_ID,
        sizeof(info.dieId), static_cast<void *>(&info.dieId));
    endpointCopy = endpoint;
    info.bwStatus = HcclRankGraphGetEndpointInfo(
        comm, ownerRank, &endpointCopy, ENDPOINT_ATTR_BW_COEFF,
        sizeof(info.bwCoeff), static_cast<void *>(&info.bwCoeff));
    endpointCopy = endpoint;
    info.locationStatus = HcclRankGraphGetEndpointInfo(
        comm, ownerRank, &endpointCopy, ENDPOINT_ATTR_LOCATION,
        sizeof(info.location), static_cast<void *>(&info.location));
    return info;
}

std::string AttrToString(uint32_t value, HcclResult status)
{
    if (status != HCCL_SUCCESS) {
        return "N/A(status=" + std::to_string(static_cast<int>(status)) + ")";
    }
    return std::to_string(value) + "(status=0)";
}

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
            const EndpointInfo srcInfo = QueryEndpointInfo(comm, rank, link.srcEndpointDesc);
            const EndpointInfo dstInfo = QueryEndpointInfo(comm, peer, link.dstEndpointDesc);
            std::printf("[A5 CCU URMA][rank=%u peer=%u] route=%u layer=%u link=%u "
                        "protocol=%d hop=%u src_phy=%u dst_phy=%u src_die=%s dst_die=%s "
                        "src_bw=%s dst_bw=%s src_location=%s dst_location=%s "
                        "src_addr=%s dst_addr=%s%s\n",
                        rank, peer, ordinal, layers[layerIndex], linkIndex,
                        static_cast<int>(link.linkAttr.linkProtocol),
                        static_cast<uint32_t>(link.linkAttr.hop),
                        link.srcEndpointDesc.loc.device.devPhyId,
                        link.dstEndpointDesc.loc.device.devPhyId,
                        AttrToString(srcInfo.dieId, srcInfo.dieStatus).c_str(),
                        AttrToString(dstInfo.dieId, dstInfo.dieStatus).c_str(),
                        AttrToString(srcInfo.bwCoeff, srcInfo.bwStatus).c_str(),
                        AttrToString(dstInfo.bwCoeff, dstInfo.bwStatus).c_str(),
                        AttrToString(srcInfo.location, srcInfo.locationStatus).c_str(),
                        AttrToString(dstInfo.location, dstInfo.locationStatus).c_str(),
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
    const EndpointInfo selectedInfo = QueryEndpointInfo(comm, rank, selectedLink.srcEndpointDesc);
    status = selectedInfo.dieStatus;
    if (status != HCCL_SUCCESS) {
        std::fprintf(stderr,
            "[A5 CCU URMA][rank=%u peer=%u] get selected endpoint die id failed: status=%d\n",
            rank, peer, static_cast<int>(status));
        return status;
    }
    *dieId = selectedInfo.dieId;
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

HcclResult GetCcuRouteIndices(std::vector<uint32_t> *routeIndices)
{
    if (routeIndices == nullptr) {
        return HCCL_E_PTR;
    }
    routeIndices->clear();
    const char *multiValue = std::getenv("A5_CCU_ROUTE_INDICES");
    if (multiValue == nullptr || multiValue[0] == '\0') {
        uint32_t routeIndex = 0;
        HcclResult status = GetCcuRouteIndex(&routeIndex);
        if (status == HCCL_SUCCESS) {
            routeIndices->push_back(routeIndex);
        }
        return status;
    }
    std::stringstream input(multiValue);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (item.empty()) {
            return HCCL_E_PARA;
        }
        errno = 0;
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(item.c_str(), &end, 10);
        if (errno != 0 || end == item.c_str() || *end != '\0' || parsed > UINT32_MAX) {
            std::fprintf(stderr, "[A5 CCU URMA] invalid route list: %s\n", multiValue);
            return HCCL_E_PARA;
        }
        const uint32_t value = static_cast<uint32_t>(parsed);
        if (std::find(routeIndices->begin(), routeIndices->end(), value) != routeIndices->end()) {
            std::fprintf(stderr, "[A5 CCU URMA] duplicate route index %u\n", value);
            return HCCL_E_PARA;
        }
        routeIndices->push_back(value);
    }
    if (routeIndices->empty() || routeIndices->size() > 8) {
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult GetRouteResources(HcclComm comm, aclrtStream stream, RouteResources *resources)
{
    if (comm == nullptr || stream == nullptr || resources == nullptr) {
        return HCCL_E_PTR;
    }
    std::vector<uint32_t> routeIndices;
    HcclResult status = GetCcuRouteIndices(&routeIndices);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    const char *routeKeyValue = std::getenv("A5_CCU_ROUTE_INDICES");
    const char *manifestValue = std::getenv("A5_CCU_SOURCE_ROUTE_MANIFEST");
    const std::string routeKey = manifestValue != nullptr && manifestValue[0] != '\0' ?
        std::string("provider:") + manifestValue :
        (routeKeyValue == nullptr ? std::to_string(routeIndices.front()) : std::string(routeKeyValue));
    if (g_cache.comm == comm && g_cache.stream == stream && g_cache.routeKey == routeKey) {
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
    std::vector<HcclChannelDesc> selectedDescs;
    std::vector<A5UvsSourceRoute> providerRoutes;
    if (manifestValue != nullptr && manifestValue[0] != '\0') {
        status = LoadProviderRoutes(comm, rank, 1U - rank, &providerRoutes);
        if (status != HCCL_SUCCESS) {
            return status;
        }
        for (const auto &route : providerRoutes) {
            selectedDescs.push_back(route.channel);
            created.routeIndices.push_back(route.routeId);
            created.weights.push_back(route.weight == 0 ? 1U : route.weight);
            if (selectedDescs.size() == 1) {
                created.dieId = route.dieId;
            } else if (route.dieId != created.dieId) {
                std::fprintf(stderr, "[A5 CCU URMA] provider routes span multiple dies; "
                             "current kernel requires one route group per die\n");
                return HCCL_E_NOT_SUPPORT;
            }
            std::printf("[A5 CCU URMA][rank=%u] provider route=%u relay_phy=%u die=%u weight=%u\n",
                        rank, route.routeId, route.relayPhyId, route.dieId,
                        route.weight == 0 ? 1U : route.weight);
        }
        routeIndices = created.routeIndices;
    } else {
        selectedDescs.resize(routeIndices.size());
        created.routeIndices = routeIndices;
        created.weights.reserve(routeIndices.size());
        for (size_t i = 0; i < routeIndices.size(); ++i) {
            uint32_t dieId = 0;
            status = SelectRoute(comm, rank, 1U - rank, routeIndices[i],
                                 &selectedDescs[i], &dieId);
            if (status != HCCL_SUCCESS) {
                return status;
            }
            if (i == 0) {
                created.dieId = dieId;
            } else if (dieId != created.dieId) {
                std::fprintf(stderr, "[A5 CCU URMA] selected routes span die %u and die %u; "
                             "split them into one CCU kernel per die\n", created.dieId, dieId);
                return HCCL_E_NOT_SUPPORT;
            }
            created.weights.push_back(routeIndices[i] == 0 ? 2U : 1U);
        }
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
                rank, 1U - rank, routeIndices.front(), created.dieId);
    std::fflush(stdout);
    const auto acquireBegin = std::chrono::steady_clock::now();
    created.channels.resize(selectedDescs.size());
    status = HcclChannelAcquire(comm, COMM_ENGINE_CCU, selectedDescs.data(),
                                static_cast<uint32_t>(selectedDescs.size()), created.channels.data());
    const auto acquireEnd = std::chrono::steady_clock::now();
    const double acquireUs = std::chrono::duration<double, std::micro>(
        acquireEnd - acquireBegin).count();
    std::printf("[A5 CCU URMA][rank=%u peer=%u] HcclChannelAcquire end: status=%d "
                "die_id=%u channel=%lu acquire_us=%.3f\n",
                rank, 1U - rank, static_cast<int>(status), created.dieId,
                static_cast<unsigned long>(status == HCCL_SUCCESS ? created.channels.front() : 0), acquireUs);
    std::fflush(stdout);
    if (status != HCCL_SUCCESS) {
        return status;
    }

    std::vector<int32_t> channelStates(created.channels.size(), -1);
    const int32_t channelStatus = HcommChannelGetStatus(created.channels.data(),
        static_cast<uint32_t>(created.channels.size()), channelStates.data());
    if (channelStatus != 0) {
        return HCCL_E_RUNTIME;
    }
    for (size_t i = 0; i < channelStates.size(); ++i) {
        std::printf("[A5 CCU URMA][rank=%u] channel[%zu] route=%u state=%d\n",
                    rank, i, created.routeIndices[i], channelStates[i]);
    }
    std::fflush(stdout);

    if (EnvEnabled("A5_CCU_CHANNEL_ONLY")) {
        std::printf("[A5 CCU URMA][rank=%u] channel-only probe passed; skip CCU kernel registration\n", rank);
        std::fflush(stdout);
        g_cache.comm = comm;
        g_cache.stream = stream;
        g_cache.routeKey = routeKey;
        g_cache.resources = created;
        *resources = created;
        return HCCL_SUCCESS;
    }

    RouteKernelArg kernelArg(created.channels, routeIndices);
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
    g_cache.routeKey = routeKey;
    g_cache.resources = created;
    *resources = created;
    return HCCL_SUCCESS;
}

} // namespace a5_ccu_urma_probe
