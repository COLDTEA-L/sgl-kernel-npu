#ifndef A5_CCU_URMA_ROUTE_PROBE_UTILS_H
#define A5_CCU_URMA_ROUTE_PROBE_UTILS_H

#include <acl/acl_rt.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace a5_ccu_urma_probe {

enum class RouteKernelKind {
    ROUTE_WRITE,
    ALLTOALL_CONCURRENT,
    ALLTOALL_SERIAL,
};

struct RouteResources {
    ThreadHandle mainThread = 0;
    ThreadHandle routeThread = 0;
    aclrtStream slaveStream = nullptr;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> routeIndices;
    std::vector<std::string> pathUids;
    std::vector<uint32_t> weights;
    uint64_t kernel = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t dieId = 0;
};

// Normalized two-rank control-plane request used by the production-facing
// explicit multipath API.  The discovered route is always channel 0 and the
// manifest contributes channels 1..N in file order.  Keeping the request
// independent of environment variables lets a future host controller install
// the same plan directly without changing the CCU data-plane kernel.
struct RoutePlanRequest {
    bool includeDiscoveredRoute = false;
    uint32_t discoveredRoute = 0;
    std::string relayManifest;
    std::vector<uint32_t> weights;
    std::string planName;
};

HcclResult GetCcuRouteIndex(uint32_t *routeIndex);

HcclResult GetCcuRouteIndices(std::vector<uint32_t> *routeIndices);

HcclResult GetRouteResources(HcclComm comm, aclrtStream stream,
                             RouteKernelKind kernelKind, RouteResources *resources,
                             const RoutePlanRequest *plan = nullptr);

} // namespace a5_ccu_urma_probe

#endif // A5_CCU_URMA_ROUTE_PROBE_UTILS_H
