#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace l0l1 {
constexpr uint64_t Alignment = 256;
constexpr uint64_t Guard = 256;
struct Slice { uint64_t offset, bytes; };
inline std::array<Slice, 2> Split(uint64_t bytes, unsigned l0Percent) {
    if (bytes < 2 * Alignment || bytes > (256ULL << 20) || bytes % Alignment ||
        l0Percent == 0 || l0Percent >= 100)
        throw std::runtime_error("bytes must be aligned to 256, in [512,256MiB]; l0-percent in [1,99]");
    const uint64_t first = (bytes * l0Percent / 100 / Alignment) * Alignment;
    if (!first || first == bytes) throw std::runtime_error("split produced an empty path");
    return {{{0, first}, {first, bytes - first}}};
}
inline std::string PairKey(unsigned rank, const std::string& local, const std::string& remote) {
    return rank == 0 ? local + ">" + remote : remote + ">" + local;
}
inline uint32_t Pattern(unsigned rank, unsigned epoch, uint64_t index) {
    // Check rank, epoch, position and both split boundaries; not constant fill.
    return uint32_t((index * 2654435761ULL) ^ ((rank + 1) * 0x13579bdfU) ^ (epoch * 0x1020304U));
}
}  // namespace l0l1
