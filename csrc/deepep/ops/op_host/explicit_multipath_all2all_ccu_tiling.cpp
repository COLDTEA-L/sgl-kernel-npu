#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "error_log.h"
#include "mc2_tiling_utils.h"
#include "register/op_def_registry.h"
#include "tiling/hccl/hccl_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/explicit_multipath_all2all_ccu_tiling.h"

namespace {
constexpr uint32_t MAX_CCU_RANKS = 32U;
constexpr size_t MAX_ATTR_LENGTH = 128UL;
constexpr size_t SYSTEM_WORKSPACE_BYTES = 16UL * 1024UL * 1024UL;

uint64_t NumElements(const gert::StorageShape *shape)
{
    uint64_t count = 1UL;
    for (size_t i = 0; i < shape->GetStorageShape().GetDimNum(); ++i) {
        count *= static_cast<uint64_t>(shape->GetStorageShape().GetDim(i));
    }
    return count;
}

uint64_t DataTypeBytes(ge::DataType type)
{
    if (type == ge::DT_FLOAT16 || type == ge::DT_BF16) return 2UL;
    if (type == ge::DT_FLOAT || type == ge::DT_INT32) return 4UL;
    return 0UL;
}

uint64_t Fnv1a64(const char *value)
{
    uint64_t hash = 1469598103934665603ULL;
    while (*value != '\0') {
        hash ^= static_cast<uint8_t>(*value++);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool ParseWeights(const char *text, std::vector<uint32_t> &weights)
{
    if (text == nullptr || text[0] == '\0') return false;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty() || weights.size() == A5_EXPLICIT_MULTIPATH_MAX_PATHS) return false;
        size_t consumed = 0;
        unsigned long value = 0;
        try {
            value = std::stoul(item, &consumed, 10);
        } catch (...) {
            return false;
        }
        if (consumed != item.size() || value == 0 || value > std::numeric_limits<uint32_t>::max()) return false;
        weights.push_back(static_cast<uint32_t>(value));
    }
    return weights.size() >= 2U;
}
}  // namespace

namespace optiling {
static ge::graphStatus ExplicitMultipathAll2AllCcuTiling(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto *tiling = context->GetTilingData<ExplicitMultipathAll2AllCcuTilingData>();
    const auto *sendShape = context->GetInputShape(0);
    const auto *sendDesc = context->GetInputDesc(0);
    const auto *recvDesc = context->GetOutputDesc(0);
    OP_TILING_CHECK(tiling == nullptr || sendShape == nullptr || sendDesc == nullptr || recvDesc == nullptr,
                    OP_LOGE(nodeName, "input/output metadata is null"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null"), return ge::GRAPH_FAILED);
    auto group = attrs->GetAttrPointer<char>(0);
    auto rankSize = attrs->GetAttrPointer<int64_t>(1);
    auto rankId = attrs->GetAttrPointer<int64_t>(2);
    auto planId = attrs->GetAttrPointer<char>(3);
    auto weightText = attrs->GetAttrPointer<char>(4);
    OP_TILING_CHECK(group == nullptr || strnlen(group, MAX_ATTR_LENGTH) == 0UL ||
                        strnlen(group, MAX_ATTR_LENGTH) == MAX_ATTR_LENGTH,
                    OP_LOGE(nodeName, "group is invalid"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(planId == nullptr || strnlen(planId, MAX_ATTR_LENGTH) == 0UL ||
                        strnlen(planId, MAX_ATTR_LENGTH) == MAX_ATTR_LENGTH,
                    OP_LOGE(nodeName, "plan_id is invalid"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankSize == nullptr || *rankSize != 2,
                    OP_LOGE(nodeName, "this revision supports exactly two ranks"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankId == nullptr || *rankId < 0 || *rankId >= *rankSize,
                    OP_LOGE(nodeName, "rank_id is invalid"), return ge::GRAPH_FAILED);

    std::vector<uint32_t> weights;
    OP_TILING_CHECK(!ParseWeights(weightText, weights),
                    OP_LOGE(nodeName, "path_weights must contain 2..8 positive integers"),
                    return ge::GRAPH_FAILED);

    const uint64_t sendCount = NumElements(sendShape);
    const uint64_t elementBytes = DataTypeBytes(sendDesc->GetDataType());
    OP_TILING_CHECK(elementBytes == 0UL || recvDesc->GetDataType() != sendDesc->GetDataType(),
                    OP_LOGE(nodeName, "unsupported or mismatched dtype"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCount == 0UL || sendCount % static_cast<uint64_t>(*rankSize) != 0UL,
                    OP_LOGE(nodeName, "sendData elements must be divisible by rank_size"),
                    return ge::GRAPH_FAILED);

    tiling->info.rankSize = static_cast<uint32_t>(*rankSize);
    tiling->info.rankId = static_cast<uint32_t>(*rankId);
    tiling->info.pathCount = static_cast<uint32_t>(weights.size());
    tiling->info.reserved = 0U;
    tiling->info.sendCount = sendCount;
    tiling->info.perRankBytes = sendCount / static_cast<uint64_t>(*rankSize) * elementBytes;
    tiling->info.planHash = Fnv1a64(planId);
    for (uint32_t i = 0; i < A5_EXPLICIT_MULTIPATH_MAX_PATHS; ++i) {
        tiling->info.pathWeights[i] = i < weights.size() ? weights[i] : 0U;
    }

    // The HCCL extension resolves plan_id while it provisions the communicator
    // resource.  The AIV side only submits the already prepared ALLTOALL task.
    const uint32_t opType = static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL);
    AscendC::Mc2CcTilingConfig config(
        std::string(group), opType, "AlltoAll=level0:fullmesh;level1:pairwise");
    config.SetCommEngine(mc2tiling::A5_CCU_ENGINE);
    config.GetTiling(tiling->mc2InitTiling);
    config.GetTiling(tiling->mc2CcTiling);

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE(nodeName, "workspace metadata is null"),
                    return ge::GRAPH_FAILED);
    workspace[0] = SYSTEM_WORKSPACE_BYTES;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aivNum = platform.GetCoreNumAiv();
    OP_TILING_CHECK(aivNum == 0U, OP_LOGE(nodeName, "no AIV core"), return ge::GRAPH_FAILED);
    context->SetTilingKey(0UL);
    context->SetBlockDim(platform.CalcTschBlockDim(aivNum, 0U, aivNum));
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

struct ExplicitMultipathAll2AllCcuCompileInfo {};
ge::graphStatus TilingParseForExplicitMultipathAll2AllCcu(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ExplicitMultipathAll2AllCcu)
    .Tiling(ExplicitMultipathAll2AllCcuTiling)
    .TilingParse<ExplicitMultipathAll2AllCcuCompileInfo>(TilingParseForExplicitMultipathAll2AllCcu);
}  // namespace optiling
