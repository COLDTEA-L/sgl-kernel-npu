#include "utils.h"
#include "route_kernel.h"
#include "all_to_all_multiroute_kernel.h"
#include "source_route_provider.h"

#include <hccl/hccl_rank_graph.h>
#include <hcomm/hcomm_res.h>
#include <hcomm/ccu/hccl_ccu_res.h>

#include <cerrno>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
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

void SetUrmaTraceLabel(const std::string &label)
{
    using SetLabelFn = void (*)(const char *);
    static SetLabelFn setLabel = reinterpret_cast<SetLabelFn>(
        dlsym(RTLD_DEFAULT, "A5UrmaTpTraceSetLabel"));
    if (setLabel != nullptr) setLabel(label.c_str());
}

struct EndpointInfo {
    uint32_t dieId = UINT32_MAX;
    uint32_t bwCoeff = UINT32_MAX;
    uint32_t location = UINT32_MAX;
    HcclResult dieStatus = HCCL_E_NOT_FOUND;
    HcclResult bwStatus = HCCL_E_NOT_FOUND;
    HcclResult locationStatus = HCCL_E_NOT_FOUND;
};

struct PathCandidate {
    CommLink link{};
    uint32_t ordinal = 0;
    uint32_t layer = 0;
    uint32_t linkIndex = 0;
    uint32_t dieId = UINT32_MAX;
    std::string uid;
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

template <typename T>
std::string RawObjectToHex(const T &object)
{
    const auto *bytes = reinterpret_cast<const unsigned char *>(&object);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < sizeof(T); ++i) {
        out << std::setw(2) << static_cast<uint32_t>(bytes[i]);
    }
    return out.str();
}

void TraceSelectedLink(uint32_t rank, uint32_t peer, const PathCandidate &selected,
                       const HcclChannelDesc &desc)
{
    if (!EnvEnabled("A5_CCU_TRACE_LINK")) return;
    const CommLink &link = selected.link;
    std::printf("COMMLINK_TRACE rank=%u peer=%u path_uid=%s ordinal=%u "
                "object_size=%zu src_endpoint_size=%zu dst_endpoint_size=%zu "
                "src_addr=%s dst_addr=%s commlink_raw=%s src_endpoint_raw=%s dst_endpoint_raw=%s\n",
                rank, peer, selected.uid.c_str(), selected.ordinal,
                sizeof(CommLink), sizeof(link.srcEndpointDesc), sizeof(link.dstEndpointDesc),
                CommAddrToString(link.srcEndpointDesc.commAddr).c_str(),
                CommAddrToString(link.dstEndpointDesc.commAddr).c_str(),
                RawObjectToHex(link).c_str(), RawObjectToHex(link.srcEndpointDesc).c_str(),
                RawObjectToHex(link.dstEndpointDesc).c_str());
    std::printf("CHANNEL_DESC_TRACE phase=before_acquire rank=%u peer=%u path_uid=%s ordinal=%u "
                "object_size=%zu remote_rank=%u notify_num=%u protocol=%d "
                "local_addr=%s remote_addr=%s channel_desc_raw=%s\n",
                rank, peer, selected.uid.c_str(), selected.ordinal, sizeof(HcclChannelDesc),
                desc.remoteRank, desc.notifyNum, static_cast<int>(desc.channelProtocol),
                CommAddrToString(desc.localEndpoint.commAddr).c_str(),
                CommAddrToString(desc.remoteEndpoint.commAddr).c_str(),
                RawObjectToHex(desc).c_str());
    std::fflush(stdout);
}

uint64_t Fnv1a64(const std::string &value)
{
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string PathUid(const CommLink &link, uint32_t layer)
{
    std::string endpointA = CommAddrToString(link.srcEndpointDesc.commAddr);
    std::string endpointB = CommAddrToString(link.dstEndpointDesc.commAddr);
    if (endpointB < endpointA) {
        std::swap(endpointA, endpointB);
    }
    const std::string identity = std::to_string(layer) + "|" +
        std::to_string(static_cast<int>(link.linkAttr.linkProtocol)) + "|" +
        std::to_string(static_cast<uint32_t>(link.linkAttr.hop)) + "|" +
        endpointA + "|" + endpointB;
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << Fnv1a64(identity);
    return out.str();
}

HcclResult EnumeratePaths(HcclComm comm, uint32_t rank, uint32_t peer,
                          std::vector<PathCandidate> *candidates)
{
    if (candidates == nullptr) return HCCL_E_PTR;
    candidates->clear();
    uint32_t layerNum = 0;
    uint32_t *layers = nullptr;
    HcclResult status = HcclRankGraphGetLayers(comm, &layers, &layerNum);
    if (status != HCCL_SUCCESS) return status;
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        uint32_t listSize = 0;
        CommLink *linkList = nullptr;
        status = HcclRankGraphGetLinks(comm, layers[layerIndex], rank, peer,
                                       &linkList, &listSize);
        if (status != HCCL_SUCCESS) return status;
        for (uint32_t linkIndex = 0; linkIndex < listSize; ++linkIndex) {
            const CommLink &link = linkList[linkIndex];
            if (link.linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) continue;
            PathCandidate candidate;
            candidate.link = link;
            candidate.ordinal = static_cast<uint32_t>(candidates->size());
            candidate.layer = layers[layerIndex];
            candidate.linkIndex = linkIndex;
            candidate.uid = PathUid(link, candidate.layer);
            const EndpointInfo srcInfo = QueryEndpointInfo(comm, rank, link.srcEndpointDesc);
            candidate.dieId = srcInfo.dieId;
            const EndpointInfo dstInfo = QueryEndpointInfo(comm, peer, link.dstEndpointDesc);
            std::printf("PATH_CATALOG rank=%u peer=%u path_uid=%s ordinal=%u layer=%u link=%u "
                        "protocol=%d hop=%u src_phy=%u dst_phy=%u src_die=%s dst_die=%s "
                        "src_bw=%s dst_bw=%s src_location=%s dst_location=%s src_addr=%s dst_addr=%s\n",
                        rank, peer, candidate.uid.c_str(), candidate.ordinal,
                        candidate.layer, candidate.linkIndex,
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
                        CommAddrToString(link.srcEndpointDesc.commAddr).c_str(),
                        CommAddrToString(link.dstEndpointDesc.commAddr).c_str());
            candidates->push_back(candidate);
        }
    }
    std::fflush(stdout);
    return candidates->empty() ? HCCL_E_NOT_FOUND : HCCL_SUCCESS;
}

HcclResult SelectRoute(HcclComm comm, uint32_t rank, uint32_t peer,
                         uint32_t routeIndex, HcclChannelDesc *desc,
                         uint32_t *dieId, std::string *pathUid)
{
    if (desc == nullptr || dieId == nullptr) {
        return HCCL_E_PTR;
    }
    std::vector<PathCandidate> candidates;
    HcclResult status = EnumeratePaths(comm, rank, peer, &candidates);
    if (status != HCCL_SUCCESS) return status;

    if (routeIndex >= candidates.size()) {
        std::fprintf(stderr,
            "[A5 CCU URMA] route index %u is out of range; rank %u -> %u has %zu UBC_CTP route(s)\n",
            routeIndex, rank, peer, candidates.size());
        return candidates.empty() ? HCCL_E_NOT_FOUND : HCCL_E_PARA;
    }

    const PathCandidate &selected = candidates[routeIndex];
    const CommLink &selectedLink = selected.link;
    const EndpointInfo selectedInfo = QueryEndpointInfo(comm, rank, selectedLink.srcEndpointDesc);
    status = selectedInfo.dieStatus;
    if (status != HCCL_SUCCESS) {
        std::fprintf(stderr,
            "[A5 CCU URMA][rank=%u peer=%u] get selected endpoint die id failed: status=%d\n",
            rank, peer, static_cast<int>(status));
        return status;
    }
    *dieId = selectedInfo.dieId;
    if (pathUid != nullptr) *pathUid = selected.uid;
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
    TraceSelectedLink(rank, peer, selected, *desc);

    std::printf("[A5 CCU URMA][rank=%u peer=%u] prepared route=%u layer=%u link=%u "
                "die_id=%u; threads will be acquired before this per-die channel\n",
                rank, peer, routeIndex, selected.layer, selected.linkIndex, *dieId);
    std::fflush(stdout);
    return HCCL_SUCCESS;
}

HcclResult ResolvePathUids(HcclComm comm, uint32_t rank, uint32_t peer,
                           const std::vector<std::string> &uids,
                           std::vector<uint32_t> *indices)
{
    if (indices == nullptr) return HCCL_E_PTR;
    std::vector<PathCandidate> candidates;
    HcclResult status = EnumeratePaths(comm, rank, peer, &candidates);
    if (status != HCCL_SUCCESS) return status;
    indices->clear();
    for (const auto &uid : uids) {
        auto found = std::find_if(candidates.begin(), candidates.end(),
            [&uid](const PathCandidate &candidate) { return candidate.uid == uid; });
        if (found == candidates.end()) {
            std::fprintf(stderr, "[A5 CCU URMA] path_uid=%s is not in this session catalog\n",
                         uid.c_str());
            return HCCL_E_NOT_FOUND;
        }
        indices->push_back(found->ordinal);
    }
    return HCCL_SUCCESS;
}

HcclResult ParseStringList(const char *name, std::vector<std::string> *values)
{
    if (values == nullptr) return HCCL_E_PTR;
    values->clear();
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') return HCCL_E_NOT_FOUND;
    std::stringstream input(raw);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (item.empty() || std::find(values->begin(), values->end(), item) != values->end()) {
            return HCCL_E_PARA;
        }
        values->push_back(item);
    }
    return values->empty() ? HCCL_E_PARA : HCCL_SUCCESS;
}

HcclResult ParseWeights(size_t count, std::vector<uint32_t> *weights)
{
    if (weights == nullptr) return HCCL_E_PTR;
    weights->assign(count, 1U);
    const char *raw = std::getenv("A5_CCU_PATH_WEIGHTS");
    if (raw == nullptr || raw[0] == '\0') return HCCL_SUCCESS;
    std::stringstream input(raw);
    std::string item;
    size_t index = 0;
    while (std::getline(input, item, ',')) {
        if (index >= count) return HCCL_E_PARA;
        char *end = nullptr;
        const unsigned long value = std::strtoul(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' || value == 0 || value > UINT32_MAX) {
            return HCCL_E_PARA;
        }
        (*weights)[index++] = static_cast<uint32_t>(value);
    }
    return index == count ? HCCL_SUCCESS : HCCL_E_PARA;
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

HcclResult GetRouteResources(HcclComm comm, aclrtStream stream,
                             RouteKernelKind kernelKind, RouteResources *resources)
{
    if (comm == nullptr || stream == nullptr || resources == nullptr) {
        return HCCL_E_PTR;
    }
    std::vector<uint32_t> routeIndices;
    HcclResult status = GetCcuRouteIndices(&routeIndices);
    if (status != HCCL_SUCCESS) {
        return status;
    }
    std::vector<std::string> requestedPathUids;
    const HcclResult uidStatus = ParseStringList("A5_CCU_PATH_UIDS", &requestedPathUids);
    if (uidStatus != HCCL_SUCCESS && uidStatus != HCCL_E_NOT_FOUND) return uidStatus;
    const char *routeKeyValue = std::getenv("A5_CCU_ROUTE_INDICES");
    const char *pathKeyValue = std::getenv("A5_CCU_PATH_UIDS");
    const char *weightKeyValue = std::getenv("A5_CCU_PATH_WEIGHTS");
    const char *manifestValue = std::getenv("A5_CCU_SOURCE_ROUTE_MANIFEST");
    std::string routeKey = manifestValue != nullptr && manifestValue[0] != '\0' ?
        std::string("provider:") + manifestValue :
        (pathKeyValue != nullptr && pathKeyValue[0] != '\0' ?
            std::string("paths:") + pathKeyValue :
            (routeKeyValue == nullptr ? std::to_string(routeIndices.front()) : std::string(routeKeyValue)));
    routeKey += ":weights=" + std::string(weightKeyValue == nullptr ? "default" : weightKeyValue);
    routeKey += ":kernel=" + std::to_string(static_cast<int>(kernelKind));
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
    if (!requestedPathUids.empty()) {
        status = ResolvePathUids(comm, rank, 1U - rank, requestedPathUids, &routeIndices);
        if (status != HCCL_SUCCESS) return status;
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
            std::string pathUid;
            status = SelectRoute(comm, rank, 1U - rank, routeIndices[i],
                                 &selectedDescs[i], &dieId, &pathUid);
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
            created.pathUids.push_back(pathUid);
        }
        status = ParseWeights(routeIndices.size(), &created.weights);
        if (status != HCCL_SUCCESS) {
            std::fprintf(stderr, "[A5 CCU URMA] A5_CCU_PATH_WEIGHTS must contain one positive integer per selected path\n");
            return status;
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
    created.channels.resize(selectedDescs.size());
    const auto acquireBegin = std::chrono::steady_clock::now();
    if (EnvEnabled("A5_CCU_SEQUENTIAL_ACQUIRE_TRACE")) {
        for (size_t i = 0; i < selectedDescs.size(); ++i) {
            const std::string uid = i < created.pathUids.size() ? created.pathUids[i] : "provider";
            std::ostringstream label;
            label << "path_uid=" << uid << ";ordinal=" << routeIndices[i]
                  << ";acquire_order=" << i;
            SetUrmaTraceLabel(label.str());
            std::printf("CHANNEL_ACQUIRE_SCOPE phase=begin rank=%u path_uid=%s ordinal=%u acquire_order=%zu\n",
                        rank, uid.c_str(), routeIndices[i], i);
            std::fflush(stdout);
            status = HcclChannelAcquire(comm, COMM_ENGINE_CCU, &selectedDescs[i], 1U,
                                        &created.channels[i]);
            std::printf("CHANNEL_ACQUIRE_SCOPE phase=end rank=%u path_uid=%s ordinal=%u "
                        "acquire_order=%zu status=%d channel_handle=%lu\n",
                        rank, uid.c_str(), routeIndices[i], i, static_cast<int>(status),
                        static_cast<unsigned long>(status == HCCL_SUCCESS ? created.channels[i] : 0));
            std::fflush(stdout);
            SetUrmaTraceLabel("");
            if (status != HCCL_SUCCESS) break;
        }
    } else {
        status = HcclChannelAcquire(comm, COMM_ENGINE_CCU, selectedDescs.data(),
                                    static_cast<uint32_t>(selectedDescs.size()), created.channels.data());
    }
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
        const char *uid = i < created.pathUids.size() ? created.pathUids[i].c_str() : "provider";
        std::printf("PATH_CHANNEL rank=%u channel_index=%zu path_uid=%s ordinal=%u weight=%u state=%d\n",
                    rank, i, uid, created.routeIndices[i], created.weights[i], channelStates[i]);
        if (EnvEnabled("A5_CCU_TRACE_LINK")) {
            std::printf("CHANNEL_HANDLE_TRACE phase=after_acquire rank=%u peer=%u "
                        "channel_index=%zu path_uid=%s ordinal=%u channel_handle=%lu state=%d\n",
                        rank, 1U - rank, i, uid, created.routeIndices[i],
                        static_cast<unsigned long>(created.channels[i]), channelStates[i]);
        }
    }
    std::fflush(stdout);

    if (EnvEnabled("A5_CCU_CHANNEL_ONLY")) {
        std::printf("[A5 CCU URMA][rank=%u] channel-only probe passed; skip CCU kernel registration\n", rank);
        std::fflush(stdout);
        const char *holdValue = std::getenv("A5_CCU_CHANNEL_HOLD_SECONDS");
        if (holdValue != nullptr && holdValue[0] != '\0') {
            char *end = nullptr;
            const unsigned long holdSeconds = std::strtoul(holdValue, &end, 10);
            if (end == holdValue || *end != '\0' || holdSeconds > 600UL) {
                std::fprintf(stderr, "[A5 CCU URMA] invalid A5_CCU_CHANNEL_HOLD_SECONDS=%s\n",
                             holdValue);
                return HCCL_E_PARA;
            }
            if (holdSeconds != 0) {
                std::printf("CHANNEL_TRACE_HOLD rank=%u seconds=%lu channel_count=%zu\n",
                            rank, holdSeconds, created.channels.size());
                std::fflush(stdout);
                std::this_thread::sleep_for(std::chrono::seconds(holdSeconds));
            }
        }
        g_cache.comm = comm;
        g_cache.stream = stream;
        g_cache.routeKey = routeKey;
        g_cache.resources = created;
        *resources = created;
        return HCCL_SUCCESS;
    }

    CcuKernelHandle kernel = 0;
    HcclResult registerStatus = HCCL_E_INTERNAL;
    const char *kernelName = "unknown";
    std::printf("[A5 CCU URMA][rank=%u] register %s begin\n", rank,
                kernelKind == RouteKernelKind::ROUTE_WRITE ? "route_write" :
                (kernelKind == RouteKernelKind::ALLTOALL_SERIAL ?
                    "alltoall_serial" : "alltoall_concurrent"));
    std::fflush(stdout);
    if (kernelKind == RouteKernelKind::ROUTE_WRITE) {
        kernelName = "route_write";
        RouteKernelArg kernelArg(created.channels, routeIndices);
        hcomm::KernelCreator creator = CreateRouteKernel;
        registerStatus = HcclCcuKernelRegister(comm, &kernel, &creator, &kernelArg);
    } else {
        const bool serialized = kernelKind == RouteKernelKind::ALLTOALL_SERIAL;
        kernelName = serialized ? "alltoall_serial" : "alltoall_concurrent";
        AllToAllMultiRouteKernelArg kernelArg(created.channels, routeIndices, serialized);
        hcomm::KernelCreator creator = CreateAllToAllMultiRouteKernel;
        registerStatus = HcclCcuKernelRegister(comm, &kernel, &creator, &kernelArg);
    }
    std::printf("[A5 CCU URMA][rank=%u] register %s end: status=%d kernel=%lu\n",
                rank, kernelName, static_cast<int>(registerStatus),
                static_cast<unsigned long>(kernel));
    std::fflush(stdout);
    if (registerStatus != HCCL_SUCCESS) {
        return registerStatus;
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
