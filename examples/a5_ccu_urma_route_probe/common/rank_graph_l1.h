#ifndef A5_CCU_RANK_GRAPH_L1_H
#define A5_CCU_RANK_GRAPH_L1_H
#include <algorithm>
#include <hccl/hccl_rank_graph.h>

namespace a5_ccu_urma_probe {
// Match graph objects, never construct an endpoint from a donor. IPv6-tagged
// EID bytes are accepted only on a UB_CTP graph link and compared byte-exactly.
inline bool MatchesGraphL1(const CommLink &link, uint32_t layer,
    const EndpointDesc &localL0, const EndpointDesc &remoteL0,
    const uint8_t *localEid, const uint8_t *remoteEid)
{
    const auto isEid = [](const EndpointDesc &e) {
        return e.commAddr.type == COMM_ADDR_TYPE_EID || e.commAddr.type == COMM_ADDR_TYPE_IP_V6;
    };
    const auto &a = link.srcEndpointDesc;
    const auto &b = link.dstEndpointDesc;
    return layer == 1 && link.linkAttr.linkProtocol == COMM_PROTOCOL_UBC_CTP &&
        a.protocol == COMM_PROTOCOL_UBC_CTP && b.protocol == COMM_PROTOCOL_UBC_CTP &&
        isEid(a) && isEid(b) &&
        a.loc.locType == localL0.loc.locType && b.loc.locType == remoteL0.loc.locType &&
        a.loc.device.devPhyId == localL0.loc.device.devPhyId &&
        b.loc.device.devPhyId == remoteL0.loc.device.devPhyId &&
        std::any_of(localEid, localEid + 16, [](uint8_t v) {return v != 0;}) &&
        std::any_of(remoteEid, remoteEid + 16, [](uint8_t v) {return v != 0;}) &&
        std::equal(localEid, localEid + 16, a.commAddr.eid) &&
        std::equal(remoteEid, remoteEid + 16, b.commAddr.eid);
}
}
#endif
