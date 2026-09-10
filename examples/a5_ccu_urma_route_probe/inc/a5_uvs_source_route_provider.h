#ifndef A5_UVS_SOURCE_ROUTE_PROVIDER_H
#define A5_UVS_SOURCE_ROUTE_PROVIDER_H

#include <hccl/hccl_comm.h>
#include <hccl/hccl_res.h>

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define A5_UVS_SOURCE_ROUTE_ABI_VERSION 1U
#define A5_UVS_MAX_SOURCE_ROUTES 8U

typedef struct A5UvsSourceRoute {
    HcclChannelDesc channel;
    uint32_t routeId;
    uint32_t relayPhyId;
    uint32_t dieId;
    uint32_t weight;
} A5UvsSourceRoute;

/*
 * Provider contract for platform UVS/HIXL integration.
 * The provider must install the forwarding rule before returning, and fill a
 * channel descriptor that HCOMM can acquire even when RankGraph did not
 * enumerate the path. Returning an ordinary destination EID without actually
 * installing a source-route/next-hop rule is invalid.
 */
typedef int (*A5UvsInstallSourceRoutesFn)(
    uint32_t abiVersion,
    HcclComm comm,
    uint32_t localRank,
    uint32_t peerRank,
    const char *manifestPath,
    A5UvsSourceRoute *routes,
    uint32_t *routeCount);

typedef void (*A5UvsReleaseSourceRoutesFn)(
    HcclComm comm,
    const A5UvsSourceRoute *routes,
    uint32_t routeCount);

#ifdef __cplusplus
}
#endif

#endif
