#ifndef A5_CCU_MULTIPATH_LAYOUT_H
#define A5_CCU_MULTIPATH_LAYOUT_H

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace a5_ccu_urma_probe {
constexpr size_t PREPARED_MAX_PATHS = 13;
constexpr uint64_t PREPARED_PATH_ALIGNMENT = 256;

struct TwoRankPathLayout {
    uint64_t peerBytes = 0;
    uint64_t totalBytes = 0;
    uint64_t selfOffset = 0;
    std::vector<uint64_t> sourceOffsets;
    std::vector<uint64_t> remoteOffsets;
    std::vector<uint64_t> pathBytes;
};

// Pure preflight shared by single-candidate baselines and prepared multipath.
// The prepared-plan API enforces its own >= 2 path contract before calling us.
inline bool BuildTwoRankPathLayout(uint64_t elementsPerPeer, uint32_t rank,
    const std::vector<uint32_t> &weights, TwoRankPathLayout *result)
{
    if (result == nullptr || rank >= 2 || elementsPerPeer == 0 ||
        elementsPerPeer > std::numeric_limits<uint64_t>::max() / 8 ||
        weights.empty() || weights.size() > PREPARED_MAX_PATHS) return false;
    uint64_t totalWeight = 0;
    for (uint32_t weight : weights) {
        if (weight == 0) return false;
        totalWeight += weight;
    }
    TwoRankPathLayout layout;
    layout.peerBytes = elementsPerPeer * 4;
    layout.totalBytes = layout.peerBytes * 2;
    layout.selfOffset = rank * layout.peerBytes;
    uint64_t assigned = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        uint64_t bytes = layout.peerBytes - assigned;
        if (i + 1 != weights.size()) {
            // FP32 payload * uint32 weight can exceed uint64 even when each
            // buffer and the final quotient fit. Keep this product exact.
            bytes = static_cast<uint64_t>(
                static_cast<__uint128_t>(layout.peerBytes) * weights[i] / totalWeight);
            bytes = bytes / PREPARED_PATH_ALIGNMENT * PREPARED_PATH_ALIGNMENT;
        }
        if (bytes == 0 || bytes > layout.peerBytes - assigned) return false;
        layout.sourceOffsets.push_back((1U - rank) * layout.peerBytes + assigned);
        layout.remoteOffsets.push_back(rank * layout.peerBytes + assigned);
        layout.pathBytes.push_back(bytes);
        assigned += bytes;
    }
    if (assigned != layout.peerBytes) return false;
    *result = std::move(layout);
    return true;
}

inline bool DisjointBufferRanges(uint64_t input, uint64_t output, uint64_t bytes)
{
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    if (input == 0 || output == 0 || bytes == 0 || input > maximum - bytes ||
        output > maximum - bytes) return false;
    return input + bytes <= output || output + bytes <= input;
}

// The vendor Append() concatenates text without separators. Explicit lengths
// and delimiters prevent [1,23] / [12,3] collisions; channels bind actual paths.
template <typename Channel>
inline std::string MakePathKernelSignature(const char *kind,
    const std::vector<uint32_t> &ordinals, const std::vector<Channel> &channels)
{
    std::ostringstream signature;
    signature << kind << ":routes=" << ordinals.size() << ':';
    for (uint32_t ordinal : ordinals) signature << ordinal << ',';
    signature << ":channels=" << channels.size() << ':';
    for (Channel channel : channels) signature << channel << ',';
    return signature.str();
}
} // namespace a5_ccu_urma_probe
#endif
