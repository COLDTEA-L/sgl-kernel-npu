#ifndef A5_LOCAL_COPY_PROBE_DATA_H
#define A5_LOCAL_COPY_PROBE_DATA_H
#include <cstddef>
#include <cstdint>
#include <cstring>
namespace a5_local_probe {
constexpr size_t GUARD = 4096;
constexpr size_t CHUNK = 4096;
constexpr size_t MAX_BYTES = 64 * 1024 * 1024;
constexpr size_t HOST_BUDGET = 256 * 1024 * 1024;
inline uint8_t Pattern(uint32_t rank, uint32_t epoch, size_t index)
{
    uint64_t value = index + 0x9e3779b97f4a7c15ULL * (epoch + 1ULL) + rank * 1000003ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<uint8_t>(value ^ (value >> 31));
}
inline bool ValidSize(size_t bytes, uint32_t queued, bool chunked)
{
    return bytes >= 256 && bytes <= MAX_BYTES && bytes % 256 == 0 &&
        (!chunked || bytes % CHUNK == 0) && queued >= 1 && queued <= 32 &&
        (bytes + 2 * GUARD) <= HOST_BUDGET / (2 * queued);
}
inline void Fill(uint8_t *buffer, size_t bytes, uint32_t rank, uint32_t epoch, bool input)
{
    std::memset(buffer, input ? 0xb7 : 0xc3, bytes + 2 * GUARD);
    for (size_t i = 0; i < bytes; ++i) {
        const auto expected = Pattern(rank, epoch, i);
        buffer[GUARD + i] = input ? expected : static_cast<uint8_t>(expected ^ 0xff);
    }
}
inline bool Verify(const uint8_t *buffer, size_t bytes, uint32_t rank, uint32_t epoch,
                   bool input, size_t *bad)
{
    const auto guard = input ? 0xb7 : 0xc3;
    for (size_t i = 0; i < bytes + 2 * GUARD; ++i) {
        const auto expected = (i < GUARD || i >= GUARD + bytes) ? guard : Pattern(rank, epoch, i - GUARD);
        if (buffer[i] != expected) { if (bad) *bad = i; return false; }
    }
    return true;
}
} // namespace a5_local_probe
#endif
