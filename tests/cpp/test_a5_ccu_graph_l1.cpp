#include "rank_graph_l1.h"
#include "peer_plan.h"
#include <cassert>
#include <iostream>
using namespace a5_ccu_urma_probe;
int main()
{
    CommLink link{};
    link.linkAttr.linkProtocol = COMM_PROTOCOL_UBC_CTP;
    link.srcEndpointDesc.protocol = link.dstEndpointDesc.protocol = COMM_PROTOCOL_UBC_CTP;
    link.srcEndpointDesc.commAddr.type = link.dstEndpointDesc.commAddr.type = COMM_ADDR_TYPE_EID;
    link.srcEndpointDesc.commAddr.eid[15] = 1;
    link.dstEndpointDesc.commAddr.eid[15] = 2;
    link.srcEndpointDesc.loc.device.devPhyId = 6;
    link.dstEndpointDesc.loc.device.devPhyId = 7;
    const auto local = link.srcEndpointDesc, remote = link.dstEndpointDesc;
    const auto matches = [&](uint32_t layer = 1) {
        return MatchesGraphL1(link, layer, local, remote, local.commAddr.eid, remote.commAddr.eid);
    };
    assert(matches());
    assert(!matches(0)); // A same-address L0 link is not proof of L1 membership.
    link.srcEndpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V6;
    assert(matches()); // Preserve SDK IPv6-tagged UB descriptor.
    link.dstEndpointDesc.commAddr.eid[15] = 3;
    assert(!matches()); // No pair matching from independently valid endpoints.
    link.dstEndpointDesc = remote;
    link.srcEndpointDesc.loc.device.devPhyId = 0;
    assert(!matches());
    link.srcEndpointDesc = local;
    link.dstEndpointDesc.protocol = static_cast<CommProtocol>(0);
    assert(!matches());
    PeerPathSpec direct;
    direct.srcRank = 0; direct.dstRank = 1; direct.srcPhy = 6; direct.dstPhy = 7;
    PeerPathSpec l1 = direct;
    l1.l1 = true; l1.l1Graph = true; l1.srcEid = "src"; l1.dstEid = "dst";
    std::string error;
    assert(ValidatePeerPlan(2,2,{direct,l1},&error));
    l1.l1 = false;
    assert(!ValidatePeerPlan(2,2,{direct,l1},&error));
    std::cout << "GRAPH_L1_MATCH_PASS\n";
}
