#include "peer_plan.h"
#include <cassert>
#include <iostream>
using namespace a5_ccu_urma_probe;

std::vector<PeerPathSpec> Make(uint32_t k, bool relay)
{
    std::vector<PeerPathSpec> rows;
    for (uint32_t a = 0; a < k; ++a) {
        for (uint32_t b = a + 1; b < k; ++b) {
            PeerPathSpec p;
            p.srcRank = a; p.dstRank = b; p.srcPhy = a; p.dstPhy = b;
            rows.push_back(p);
            if (relay) {
                p.relay = true; p.weight = 1;
                // Edge coloring of K4: disjoint peer pairs share an external relay.
                p.relayPhy = k == 2 ? 4 : 3 + (a ^ b);
                p.srcEid = "0000:0000:0001:0000:0000:0000:0000:0000";
                p.dstEid = "0000:0000:0002:0000:0000:0000:0000:0000";
                rows.push_back(p);
            }
        }
    }
    return rows;
}

int main()
{
    std::string error;
    for (uint32_t k : {2U, 4U}) {
        for (bool relay : {false, true}) {
            auto rows = Make(k, relay);
            assert(ValidatePeerPlan(k, 8, rows, &error));
            for (uint32_t rank = 0; rank < k; ++rank) {
                std::vector<uint32_t> peers, weights;
                for (uint32_t peer = 0; peer < k; ++peer) {
                    if (rank == peer) continue;
                    peers.push_back(peer); weights.push_back(2);
                    if (relay) { peers.push_back(peer); weights.push_back(1); }
                }
                PeerPathLayout out;
                assert(BuildPeerPathLayout(1048576, rank, k, peers, weights, &out));
                assert(out.totalBytes == 4194304ULL * k);
                assert(out.selfOffset == rank * 4194304ULL);
                std::map<uint32_t, uint64_t> assigned;
                for (size_t i = 0; i < peers.size(); ++i) {
                    assert(out.sourceOffsets[i] == peers[i] * out.peerBytes + assigned[peers[i]]);
                    assert(out.remoteOffsets[i] == rank * out.peerBytes + assigned[peers[i]]);
                    assigned[peers[i]] += out.pathBytes[i];
                }
                for (auto entry : assigned) assert(entry.second == out.peerBytes);
                weights[0] = 0;
                assert(!BuildPeerPathLayout(1048576, rank, k, peers, weights, &out));
            }
        }
    }
    auto bad = Make(4, true);
    bad[1].relayPhy = 0;
    assert(!ValidatePeerPlan(4, 8, bad, &error));
    bad = Make(4, true);
    bad[3].relayPhy = bad[1].relayPhy;
    assert(!ValidatePeerPlan(4, 8, bad, &error));
    bad = Make(4, false); bad.pop_back();
    assert(!ValidatePeerPlan(4, 8, bad, &error));
    bad = Make(2, true);
    for (uint32_t relay : {2U, 3U, 5U, 6U, 7U}) {
        // Fill all six legal external relays exactly once.
        auto row = bad[1]; row.relayPhy = relay;
        bad.push_back(row);
    }
    assert(ValidatePeerPlan(2, 8, bad, &error));
    auto seventh = bad.back(); seventh.relayPhy = 8; bad.push_back(seventh);
    assert(!ValidatePeerPlan(2, 8, bad, &error));
    PeerPathLayout out;
    assert(!BuildPeerPathLayout(UINT64_MAX, 0, 2, {1}, {2}, &out));
    assert(!BuildPeerPathLayout(1, 0, 2, {1, 1}, {2, 1}, &out));
    assert(!BuildPeerPathLayout(1024, 0, 4, {1, 2}, {2, 2}, &out));
    assert(!BuildPeerPathLayout(1024, 0, 2, {0}, {2}, &out));
    std::cout << "2/4-rank direct-only, relays, partition/overflow/invalid plan: PASS\n";
}
