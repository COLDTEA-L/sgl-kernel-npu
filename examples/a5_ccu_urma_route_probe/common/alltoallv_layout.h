#ifndef A5_CCU_ALLTOALLV_LAYOUT_H
#define A5_CCU_ALLTOALLV_LAYOUT_H
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace a5_ccu_urma_probe {
constexpr uint32_t ALLTOALLV_RANKS = 8;
struct AllToAllVLayout {
    uint64_t sendBytes = 0, recvBytes = 0;
    uint64_t selfSourceOffset = 0, selfDestinationOffset = 0, selfBytes = 0;
    std::vector<uint64_t> sourceOffsets, remoteOffsets, pathBytes;
};

// counts[s * 8 + d] counts contiguous rows, including self. All ranks must
// supply the SAME matrix for this collective. No padding or rank deduplication.
inline bool BuildAllToAllVLayout(uint32_t rank, const uint64_t *counts,
    uint32_t count, uint64_t rowBytes, const std::vector<uint32_t> &peers,
    AllToAllVLayout *out)
{
    if (!out || !counts || rank >= ALLTOALLV_RANKS || count != 64 || !rowBytes || peers.size() != 7)
        return false;
    std::array<uint64_t, 8> sends{}, recvs{};
    std::array<uint64_t, 64> source{}, destination{}, bytes{};
    for (uint32_t s = 0; s < 8; ++s) {
        for (uint32_t d = 0; d < 8; ++d) {
            const size_t i = s * 8 + d;
            if (counts[i] > UINT64_MAX / rowBytes) return false;
            bytes[i] = counts[i] * rowBytes;
            source[i] = sends[s]; destination[i] = recvs[d];
            if (bytes[i] > UINT64_MAX - sends[s] || bytes[i] > UINT64_MAX - recvs[d]) return false;
            sends[s] += bytes[i]; recvs[d] += bytes[i];
        }
    }
    AllToAllVLayout result;
    result.sendBytes = sends[rank]; result.recvBytes = recvs[rank];
    const auto self = rank * 8 + rank;
    result.selfSourceOffset = source[self]; result.selfDestinationOffset = destination[self];
    result.selfBytes = bytes[self];
    std::array<bool, 8> seen{};
    for (auto peer : peers) {
        if (peer >= 8 || peer == rank || seen[peer]) return false;
        seen[peer] = true;
        const auto i = rank * 8 + peer;
        result.sourceOffsets.push_back(source[i]);
        result.remoteOffsets.push_back(destination[i]);
        result.pathBytes.push_back(bytes[i]);
    }
    *out = result;
    return true;
}

inline bool DisjointAllToAllVBuffers(uint64_t send, uint64_t sendCapacity,
    uint64_t recv, uint64_t recvCapacity, const AllToAllVLayout &layout)
{
    // Even logically empty ranks need valid address/token backing for metadata.
    if (!send || !recv || !sendCapacity || !recvCapacity ||
        sendCapacity < layout.sendBytes || recvCapacity < layout.recvBytes ||
        sendCapacity > UINT64_MAX - send || recvCapacity > UINT64_MAX - recv) return false;
    return send + sendCapacity <= recv || recv + recvCapacity <= send;
}
} // namespace a5_ccu_urma_probe
#endif
