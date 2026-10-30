#include "../common/eid_register_stage.h"
#ifndef A5_CCU_URMA_ROUTE_PROBE_UTILS_H
#define A5_CCU_URMA_ROUTE_PROBE_UTILS_H

#include <acl/acl_rt.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "../common/peer_plan.h"
#include "../common/eid_alltoallv_layout.h"
#include "../common/channel_sync_policy.h"

namespace a5_ccu_urma_probe {
class SelfCopyLane;

enum class RouteKernelKind {
    ROUTE_WRITE,
    ALLTOALL_CONCURRENT,
    ALLTOALL_SERIAL,
    COMMAND_BLOCK_WORKER,
    ROUTE_WRITE_LOCAL,
    ROUTE_WRITE_GROUP_LOCAL,
};

struct RouteResources {
    bool ccuSelfCopy = false; // frozen when registered, never changed per execute
    bool ccuGroupCopy = false;
    std::shared_ptr<SelfCopyLane> selfCopy;
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

struct PeerPlanResources {
    RouteResources route;
    std::vector<uint32_t> peers;
    std::vector<EidChannel> eidChannels;
    std::array<uint8_t, 64> eidAllowed{};
};

HcclResult GetPeerPlanResources(HcclComm comm, aclrtStream stream,
    const std::vector<PeerPathSpec> &specs, PeerPlanResources *resources,
    ChannelSyncMode syncMode = ChannelSyncMode::BASELINE, bool allToAllV = false, bool eidAllToAllV = false,
    EidRegisterStage eidStage = EidRegisterStage::FULL);

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
    uint64_t commandBlockAddress = 0;
};

HcclResult GetCcuRouteIndex(uint32_t *routeIndex);

HcclResult GetCcuRouteIndices(std::vector<uint32_t> *routeIndices);

HcclResult GetRouteResources(HcclComm comm, aclrtStream stream,
                             RouteKernelKind kernelKind, RouteResources *resources,
                             const RoutePlanRequest *plan = nullptr);

} // namespace a5_ccu_urma_probe

#endif // A5_CCU_URMA_ROUTE_PROBE_UTILS_H
