#ifndef A5_CCU_CHANNEL_SYNC_POLICY_H
#define A5_CCU_CHANNEL_SYNC_POLICY_H
#include <cstdlib>
#include <cstring>
namespace a5_ccu_urma_probe {
enum class ChannelSyncMode { BASELINE, PACKED, PACKED_READY };
inline const char *ChannelSyncName(ChannelSyncMode mode)
{
    switch (mode) {
        case ChannelSyncMode::BASELINE: return "baseline";
        case ChannelSyncMode::PACKED: return "packed";
        case ChannelSyncMode::PACKED_READY: return "packed-ready";
    }
    return "invalid";
}
inline bool ReadChannelSyncMode(ChannelSyncMode *mode)
{
    const char *text = std::getenv("A5_CCU_CHANNEL_SYNC_MODE");
    if (!text || std::strcmp(text, "baseline") == 0) *mode = ChannelSyncMode::BASELINE;
    else if (std::strcmp(text, "packed") == 0) *mode = ChannelSyncMode::PACKED;
    else if (std::strcmp(text, "packed-ready") == 0) *mode = ChannelSyncMode::PACKED_READY;
    else return false;
    return true;
}
} // namespace a5_ccu_urma_probe
#endif
