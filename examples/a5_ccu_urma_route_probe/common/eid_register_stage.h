#pragma once
#include <cstring>
#include <cstdlib>

namespace a5_ccu_urma_probe {
enum class EidRegisterStage { MINIMAL, ARGS, MASK, CHANNEL_VARS, SYNC, DIRECT, RELAY, SELF, FULL, MASK_INIT, MASK_NEGATIVE, MASK_COMPARE, MASK_ADD, MASK_STATIC, MASK_EQUAL, MASK_NOT_EQUAL, MASK_COMPARE_ZERO, MASK_COMPARE_SINGLE };
inline const char* EidRegisterStageName(EidRegisterStage stage) {
    static const char* names[] = {"minimal", "args", "mask", "channel_vars", "sync", "direct", "relay", "self", "full", "mask_init", "mask_negative", "mask_compare", "mask_add", "mask_static", "mask_equal", "mask_not_equal", "mask_compare_zero", "mask_compare_single"};
    const auto value = static_cast<unsigned>(stage);
    return value < 18 ? names[value] : "invalid";
}
inline bool ParseEidRegisterStage(const char* value, EidRegisterStage* stage) {
    if (!stage) return false;
    if (!value || !*value) { *stage = EidRegisterStage::FULL; return true; }
    for (unsigned i = 0; i < 18; ++i) {
        auto candidate = static_cast<EidRegisterStage>(i);
        if (std::strcmp(value, EidRegisterStageName(candidate)) == 0) { *stage = candidate; return true; }
    }
    return false;
}
inline bool ReadEidRegisterStage(EidRegisterStage* stage) {
    return ParseEidRegisterStage(std::getenv("A5_EID_REGISTER_STAGE"), stage);
}
inline bool EidStageAllowsPayload(EidRegisterStage stage) { return stage == EidRegisterStage::FULL; }
}
