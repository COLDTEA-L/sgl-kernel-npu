#ifndef A5_CCU_PEER_PLAN_H
#define A5_CCU_PEER_PLAN_H

#include "path_layout.h"
#include <map>
#include <set>

namespace a5_ccu_urma_probe {
// Canonical unordered peer pair: srcRank < dstRank. Reverse descriptors use
// the same entry with exchanged endpoint EIDs/dies. No rank-0 special case.
struct PeerPathSpec {
    uint32_t srcRank = 0, dstRank = 0, srcPhy = 0, dstPhy = 0;
    bool relay = false;
    uint32_t relayPhy = UINT32_MAX, directRoute = 0, weight = 2;
    uint32_t srcDie = 0, dstDie = 0;
    std::string srcEid, dstEid;
};

inline bool ValidatePeerPlan(uint32_t rankSize, uint32_t availableCards,
    const std::vector<PeerPathSpec> &specs, std::string *error)
{
    auto fail = [error](const char *text) {
        if (error != nullptr) *error = text;
        return false;
    };
    if ((rankSize != 2 && rankSize != 4) || availableCards < rankSize || specs.empty())
        return fail("initial peer-plan supports rankSize 2 or 4 and N >= k");
    std::map<uint32_t, uint32_t> mapping;
    std::map<std::pair<uint32_t, uint32_t>, std::vector<const PeerPathSpec *>> pairs;
    for (const auto &p : specs) {
        if (p.srcRank >= p.dstRank || p.dstRank >= rankSize || p.srcPhy == p.dstPhy ||
            p.weight == 0 || p.srcDie > 1 || p.dstDie > 1)
            return fail("invalid canonical rank pair, weight, physical IDs or endpoint die");
        for (auto endpoint : {std::make_pair(p.srcRank, p.srcPhy),
                              std::make_pair(p.dstRank, p.dstPhy)}) {
            const auto inserted = mapping.emplace(endpoint);
            if (!inserted.second && inserted.first->second != endpoint.second)
                return fail("physical device mapping differs between peer rows");
        }
        pairs[{p.srcRank, p.dstRank}].push_back(&p);
    }
    std::set<uint32_t> members;
    for (auto entry : mapping) members.insert(entry.second);
    if (mapping.size() != rankSize || members.size() != rankSize)
        return fail("rank-to-physical-device mapping must be complete and unique");
    const size_t relayLimit = (availableCards - rankSize) / (rankSize - 1);
    std::map<uint32_t, std::set<uint32_t>> usedByRank;
    std::set<uint32_t> allRelays;
    for (uint32_t a = 0; a < rankSize; ++a) {
        for (uint32_t b = a + 1; b < rankSize; ++b) {
            const auto found = pairs.find({a, b});
            if (found == pairs.end()) return fail("every unordered peer pair needs a direct row");
            const auto &paths = found->second;
            if (paths.front()->relay) return fail("first path for each peer must be direct");
            std::set<uint32_t> relays;
            size_t directs = 0;
            for (const auto *path : paths) {
                if (!path->relay) {
                    ++directs;
                } else {
                    if (members.count(path->relayPhy) || path->relayPhy == UINT32_MAX ||
                        !relays.insert(path->relayPhy).second || path->srcEid.empty() ||
                        path->dstEid.empty()) return fail("relay must be explicit, unique and outside communicator");
                    if (!usedByRank[a].insert(path->relayPhy).second ||
                        !usedByRank[b].insert(path->relayPhy).second)
                        return fail("one rank cannot reuse the same relay for different peers");
                    allRelays.insert(path->relayPhy);
                }
            }
            if (directs != 1 || relays.size() > relayLimit)
                return fail("one direct per peer; relay count exceeds (N-k)/(k-1)");
        }
    }
    if (allRelays.size() > availableCards - rankSize)
        return fail("relay device union exceeds available external cards");
    return true;
}

struct PeerPathLayout {
    uint64_t peerBytes = 0, totalBytes = 0, selfOffset = 0;
    std::vector<uint64_t> sourceOffsets, remoteOffsets, pathBytes;
};

// peers/weights are flattened in the resource's channel order. Each peer owns
// its own full byte budget, not a share of all remote peers' bytes.
inline bool BuildPeerPathLayout(uint64_t elementsPerPeer, uint32_t rank,
    uint32_t rankSize, const std::vector<uint32_t> &peers,
    const std::vector<uint32_t> &weights, PeerPathLayout *result)
{
    if (result == nullptr || (rankSize != 2 && rankSize != 4) || rank >= rankSize ||
        elementsPerPeer == 0 || elementsPerPeer > UINT64_MAX / (4U * rankSize) ||
        peers.empty() || peers.size() != weights.size() || 4 + 3 * peers.size() > 48)
        return false;
    PeerPathLayout layout;
    layout.peerBytes = elementsPerPeer * 4;
    layout.totalBytes = layout.peerBytes * rankSize;
    layout.selfOffset = rank * layout.peerBytes;
    std::set<uint32_t> seen;
    for (size_t begin = 0; begin < peers.size();) {
        const uint32_t peer = peers[begin];
        if (peer >= rankSize || peer == rank || !seen.insert(peer).second) return false;
        size_t end = begin;
        uint64_t totalWeight = 0;
        while (end < peers.size() && peers[end] == peer) {
            if (weights[end] == 0) return false;
            totalWeight += weights[end++];
        }
        uint64_t assigned = 0;
        for (size_t i = begin; i < end; ++i) {
            uint64_t bytes = layout.peerBytes - assigned;
            if (i + 1 != end) {
                bytes = static_cast<uint64_t>(static_cast<__uint128_t>(layout.peerBytes) *
                    weights[i] / totalWeight);
                bytes = bytes / PREPARED_PATH_ALIGNMENT * PREPARED_PATH_ALIGNMENT;
            }
            if (bytes == 0 || bytes > layout.peerBytes - assigned) return false;
            layout.sourceOffsets.push_back(peer * layout.peerBytes + assigned);
            layout.remoteOffsets.push_back(rank * layout.peerBytes + assigned);
            layout.pathBytes.push_back(bytes);
            assigned += bytes;
        }
        if (assigned != layout.peerBytes) return false;
        begin = end;
    }
    if (seen.size() != rankSize - 1) return false;
    *result = std::move(layout);
    return true;
}
} // namespace a5_ccu_urma_probe
#endif
