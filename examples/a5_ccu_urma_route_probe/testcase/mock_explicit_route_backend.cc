#include "a5_uvs_explicit_route_backend.h"

#include <cstring>

namespace {
uint32_t g_relayPhyId = 0;
uint32_t g_dieId = 0;
uint32_t g_srcPhyId = 0;
uint32_t g_dstPhyId = 0;
}

extern "C" int A5UvsBackendGetCapabilities(
    uint32_t abiVersion, A5UvsExplicitRouteBackendCaps *caps)
{
    if (abiVersion != A5_UVS_EXPLICIT_ROUTE_BACKEND_ABI_VERSION || caps == nullptr) {
        return -1;
    }
    caps->abiVersion = abiVersion;
    caps->capabilityFlags = A5_UVS_CAP_INSTALL_SOURCE_ROUTE |
        A5_UVS_CAP_QUERY_SOURCE_ROUTE | A5_UVS_CAP_REMOVE_SOURCE_ROUTE |
        A5_UVS_CAP_IO_DIE_FORWARDING | A5_UVS_CAP_NO_RELAY_HBM;
    caps->maxRoutes = 8;
    return 0;
}

extern "C" int A5UvsBackendInstallRoute(uint32_t abiVersion, HcclComm,
    const A5UvsExplicitRouteRequest *request, A5UvsExplicitRouteResult *result)
{
    if (abiVersion != A5_UVS_EXPLICIT_ROUTE_BACKEND_ABI_VERSION ||
        request == nullptr || result == nullptr) {
        return -1;
    }
    std::memset(result, 0, sizeof(*result));
    result->channel.remoteRank = request->peerRank;
    result->channel.channelProtocol = COMM_PROTOCOL_UBC_CTP;
    result->leaseId = 100000U + request->routeId;
    result->installedSrcPhyId = request->srcPhyId;
    result->installedDstPhyId = request->dstPhyId;
    result->installedRelayPhyId = request->relayPhyId;
    result->installedDieId = request->dieId;
    result->forwardingOnly = 1;
    g_relayPhyId = request->relayPhyId;
    g_dieId = request->dieId;
    g_srcPhyId = request->srcPhyId;
    g_dstPhyId = request->dstPhyId;
    return 0;
}

extern "C" int A5UvsBackendQueryRoute(uint32_t abiVersion, uint64_t leaseId,
    A5UvsExplicitRouteResult *result)
{
    if (abiVersion != A5_UVS_EXPLICIT_ROUTE_BACKEND_ABI_VERSION ||
        leaseId == 0 || result == nullptr) {
        return -1;
    }
    result->leaseId = leaseId;
    result->installedSrcPhyId = g_srcPhyId;
    result->installedDstPhyId = g_dstPhyId;
    result->installedRelayPhyId = g_relayPhyId;
    result->installedDieId = g_dieId;
    result->forwardingOnly = 1;
    return 0;
}

extern "C" int A5UvsBackendRemoveRoute(uint32_t abiVersion, uint64_t leaseId)
{
    return abiVersion == A5_UVS_EXPLICIT_ROUTE_BACKEND_ABI_VERSION && leaseId != 0 ? 0 : -1;
}
