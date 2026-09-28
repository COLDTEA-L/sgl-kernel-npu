#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "error_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/ccu_hbm_command_puncture_tiling.h"

namespace {
uint64_t NumElements(const gert::StorageShape *shape)
{
    uint64_t count = 1;
    for (size_t i = 0; i < shape->GetStorageShape().GetDimNum(); ++i) {
        count *= static_cast<uint64_t>(shape->GetStorageShape().GetDim(i));
    }
    return count;
}

bool ParseWeights(const char *text, std::vector<uint32_t> &weights)
{
    if (text == nullptr || text[0] == '\0') return false;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty() || weights.size() == A5_COMMAND_PUNCTURE_MAX_PATHS) return false;
        size_t consumed = 0;
        unsigned long value = 0;
        try {
            value = std::stoul(item, &consumed, 10);
        } catch (...) {
            return false;
        }
        if (consumed != item.size() || value == 0 ||
            value > std::numeric_limits<uint32_t>::max()) return false;
        weights.push_back(static_cast<uint32_t>(value));
    }
    return weights.size() >= 2U;
}
} // namespace

namespace optiling {
static ge::graphStatus CcuHbmCommandPunctureTiling(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto *tiling = context->GetTilingData<CcuHbmCommandPunctureTilingData>();
    const auto *sendShape = context->GetInputShape(0);
    const auto *recvShape = context->GetInputShape(1);
    const auto *commandShape = context->GetInputShape(2);
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(tiling == nullptr || sendShape == nullptr || recvShape == nullptr ||
                        commandShape == nullptr || attrs == nullptr,
                    OP_LOGE(nodeName, "missing input metadata"), return ge::GRAPH_FAILED);
    const auto rankId = attrs->GetAttrPointer<int64_t>(0);
    const auto weightsText = attrs->GetAttrPointer<char>(1);
    const auto transfer = attrs->GetAttrPointer<bool>(2);
    OP_TILING_CHECK(rankId == nullptr || *rankId < 0 || *rankId >= 2 || transfer == nullptr,
                    OP_LOGE(nodeName, "invalid rank_id/transfer"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(NumElements(sendShape) == 0 || NumElements(sendShape) % 2U != 0 ||
                        NumElements(sendShape) != NumElements(recvShape),
                    OP_LOGE(nodeName, "send/recv must have equal non-empty two-rank shape"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(NumElements(commandShape) < 38U,
                    OP_LOGE(nodeName, "commandBlock requires at least 38 int64 words"),
                    return ge::GRAPH_FAILED);
    std::vector<uint32_t> weights;
    OP_TILING_CHECK(!ParseWeights(weightsText, weights),
                    OP_LOGE(nodeName, "path_weights requires 2..8 positive integers"),
                    return ge::GRAPH_FAILED);
    tiling->info.rankId = static_cast<uint32_t>(*rankId);
    tiling->info.pathCount = static_cast<uint32_t>(weights.size());
    tiling->info.transfer = *transfer ? 1U : 0U;
    tiling->info.reserved = 0U;
    tiling->info.perPeerBytes = NumElements(sendShape) / 2U * sizeof(float);
    tiling->info.maxSpinCount = 1000000000ULL;
    for (uint32_t i = 0; i < A5_COMMAND_PUNCTURE_MAX_PATHS; ++i) {
        tiling->info.pathWeights[i] = i < weights.size() ? weights[i] : 0U;
    }
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE(nodeName, "workspace metadata is null"),
                    return ge::GRAPH_FAILED);
    workspace[0] = 0;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetTilingKey(0UL);
    context->SetBlockDim(platform.CalcTschBlockDim(1U, 0U, 1U));
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

struct CcuHbmCommandPunctureCompileInfo {};
ge::graphStatus TilingParseForCcuHbmCommandPuncture(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CcuHbmCommandPuncture)
    .Tiling(CcuHbmCommandPunctureTiling)
    .TilingParse<CcuHbmCommandPunctureCompileInfo>(TilingParseForCcuHbmCommandPuncture);
} // namespace optiling
