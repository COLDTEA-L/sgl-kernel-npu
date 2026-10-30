#ifndef A5_CCU_EID_ALLTOALLV_LAYOUT_H
#define A5_CCU_EID_ALLTOALLV_LAYOUT_H
#include "alltoallv_layout.h"
#include <algorithm>

namespace a5_ccu_urma_probe {
constexpr size_t EP8_EID_MAX_CHANNELS = 49;
// One end-to-end channel per (peer, relay rank); -1 denotes the direct path.
struct EidChannel { uint32_t peer; int32_t relay; };
struct EidAllToAllVLayout : AllToAllVLayout {
    // src offset, destination offset, original bytes, suffix bytes, relay rank.
    std::array<std::array<uint64_t, 5>, 7> peerArgs{};
    uint64_t activeMask = 0;
};
inline bool ValidateEidChannels(uint32_t rank, const std::vector<EidChannel>& channels) {
    if (rank >= 8 || channels.size() < 7 || channels.size() > EP8_EID_MAX_CHANNELS) return false;
    std::array<uint32_t, 8> masks{}, directs{};
    for (const auto& c : channels) {
        if (c.peer >= 8 || c.peer == rank || c.relay < -1 || c.relay >= 8 ||
            c.relay == static_cast<int32_t>(rank) || c.relay == static_cast<int32_t>(c.peer)) return false;
        if (c.relay < 0) ++directs[c.peer];
        else { if (masks[c.peer] & (1u << c.relay)) return false; masks[c.peer] |= 1u << c.relay; }
    }
    for (uint32_t p = 0; p < 8; ++p) if (p != rank && directs[p] != 1) return false;
    return true;
}
inline bool ValidateEidPolicy(const uint64_t* counts, const int32_t* relays, const uint64_t* moved,
                              uint32_t count, const std::array<uint8_t, 64>& allowed) {
    if (!counts || !relays || !moved || count != 64) return false;
    for (size_t f = 0; f < 64; ++f) {
        if (relays[f] < -1 || relays[f] >= 8 || moved[f] > counts[f]) return false;
        if (moved[f]) {
            if (f/8 == f%8 || relays[f] < 0 || !(allowed[f] & (1u << relays[f]))) return false;
        } else if (relays[f] != -1) return false;
    }
    return true;
}
inline bool BuildEidAllToAllVLayout(uint32_t rank, const uint64_t* counts, uint32_t count,
    uint64_t rowBytes, const int32_t* relays, const uint64_t* moved,
    const std::array<uint8_t, 64>& allowed, const std::vector<EidChannel>& channels,
    EidAllToAllVLayout* out) {
    if (!out || !ValidateEidChannels(rank, channels) || !ValidateEidPolicy(counts, relays, moved, count, allowed)) return false;
    std::vector<uint32_t> peers;
    for (uint32_t p = 0; p < 8; ++p) if (p != rank) peers.push_back(p);
    AllToAllVLayout base;
    if (!BuildAllToAllVLayout(rank, counts, count, rowBytes, peers, &base)) return false;
    EidAllToAllVLayout result;
    static_cast<AllToAllVLayout&>(result) = base;
    result.sourceOffsets.clear(); result.remoteOffsets.clear(); result.pathBytes.clear();
    for (size_t j = 0; j < 7; ++j) {
        auto f = rank*8+peers[j];
        result.peerArgs[j] = {base.sourceOffsets[j], base.remoteOffsets[j], base.pathBytes[j],
                             moved[f]*rowBytes, static_cast<uint64_t>(relays[f])};
    }
    for (size_t i = 0; i < channels.size(); ++i) {
        const auto& c = channels[i]; size_t f = rank*8+c.peer, reverse = c.peer*8+rank;
        size_t j = c.peer < rank ? c.peer : c.peer-1;
        const auto& a = result.peerArgs[j];
        uint64_t direct = a[2]-a[3], offset = c.relay < 0 ? 0 : direct;
        uint64_t bytes = c.relay < 0 ? direct : (relays[f] == c.relay ? a[3] : 0);
        uint64_t incoming = c.relay < 0 ? counts[reverse]-moved[reverse] : (relays[reverse] == c.relay ? moved[reverse] : 0);
        if (bytes || incoming) result.activeMask |= uint64_t{1} << i;
        result.sourceOffsets.push_back(a[0]+offset); result.remoteOffsets.push_back(a[1]+offset);
        result.pathBytes.push_back(bytes);
    }
    // The caller's candidate mask must not advertise a missing local channel.
    for (uint32_t p : peers) for (size_t f : {size_t(rank*8+p), size_t(p*8+rank)}) if (moved[f]) {
        if (std::none_of(channels.begin(), channels.end(), [&](const auto& c) {return c.peer == p && c.relay == relays[f];})) return false;
    }
    *out = std::move(result); return true;
}
} // namespace a5_ccu_urma_probe
#endif
