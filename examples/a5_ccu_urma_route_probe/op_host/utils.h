#ifndef A5_CCU_URMA_ROUTE_PROBE_UTILS_H
#define A5_CCU_URMA_ROUTE_PROBE_UTILS_H

#include <acl/acl_rt.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include <cstdint>

namespace a5_ccu_urma_probe {

struct RouteResources {
    ThreadHandle mainThread = 0;
    ThreadHandle routeThread = 0;
    aclrtStream slaveStream = nullptr;
    ChannelHandle channel = 0;
    uint64_t kernel = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t dieId = 0;
};

HcclResult GetCcuRouteIndex(uint32_t *routeIndex);

HcclResult GetRouteResources(HcclComm comm, aclrtStream stream, RouteResources *resources);

} // namespace a5_ccu_urma_probe

#endif // A5_CCU_URMA_ROUTE_PROBE_UTILS_H
