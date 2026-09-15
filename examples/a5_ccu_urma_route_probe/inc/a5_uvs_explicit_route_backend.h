#ifndef A5_UVS_EXPLICIT_ROUTE_BACKEND_H
#define A5_UVS_EXPLICIT_ROUTE_BACKEND_H

#include <hccl/hccl_comm.h>
#include <hccl/hccl_res.h>

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define A5_UVS_EXPLICIT_ROUTE_BACKEND_ABI_VERSION 1U

enum A5UvsExplicitRouteCapability {
    A5_UVS_CAP_INSTALL_SOURCE_ROUTE = 1U << 0,
    A5_UVS_CAP_QUERY_SOURCE_ROUTE = 1U << 1,
    A5_UVS_CAP_REMOVE_SOURCE_ROUTE = 1U << 2,
    A5_UVS_CAP_IO_DIE_FORWARDING = 1U << 3,
    A5_UVS_CAP_NO_RELAY_HBM = 1U << 4,
};

typedef struct A5UvsExplicitRouteBackendCaps {
    uint32_t abiVersion;
    uint32_t capabilityFlags;
    uint32_t maxRoutes;
    uint32_t reserved;
} A5UvsExplicitRouteBackendCaps;

typedef struct A5UvsExplicitRouteRequest {
    uint32_t routeId;
    uint32_t localRank;
    uint32_t peerRank;
    uint32_t srcPhyId;
    uint32_t dstPhyId;
    uint32_t relayPhyId;
    uint32_t dieId;
    uint32_t weight;
    uint32_t requireIoDieForwarding;
    uint32_t forbidRelayHbm;
} A5UvsExplicitRouteRequest;

typedef struct A5UvsExplicitRouteResult {
    HcclChannelDesc channel;
    uint64_t leaseId;
    uint32_t installedSrcPhyId;
    uint32_t installedDstPhyId;
    uint32_t installedRelayPhyId;
    uint32_t installedDieId;
    uint32_t forwardingOnly;
    uint32_t reserved;
} A5UvsExplicitRouteResult;

/* Implemented by a platform UVS/HIXL control-plane plugin. */
typedef int (*A5UvsBackendGetCapabilitiesFn)(
    uint32_t abiVersion, A5UvsExplicitRouteBackendCaps *caps);

typedef int (*A5UvsBackendInstallRouteFn)(
    uint32_t abiVersion, HcclComm comm,
    const A5UvsExplicitRouteRequest *request,
    A5UvsExplicitRouteResult *result);

typedef int (*A5UvsBackendQueryRouteFn)(
    uint32_t abiVersion, uint64_t leaseId,
    A5UvsExplicitRouteResult *result);

typedef int (*A5UvsBackendRemoveRouteFn)(
    uint32_t abiVersion, uint64_t leaseId);

#ifdef __cplusplus
}
#endif

#endif
