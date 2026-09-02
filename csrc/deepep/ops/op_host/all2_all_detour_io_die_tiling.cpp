#include <cstdint>
#include <cstring>
#include <string>

#include "error_log.h"
#include "mc2_tiling_utils.h"
#include "register/op_def_registry.h"
#include "tiling/hccl/hccl_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/all2_all_detour_io_die_tiling.h"

namespace {
constexpr uint32_t MAX_MTE_RANKS = 32U;
constexpr uint64_t A5_MTE_STATE_WIN_SIZE = 4UL * 1024UL * 1024UL;
constexpr uint64_t CELL_FLAG_BYTES = 4096UL;
constexpr uint64_t CELL_ALIGN_BYTES = 512UL;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr size_t SYSTEM_WORKSPACE_BYTES = 16UL * 1024UL * 1024UL;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1UL) / alignment * alignment;
}

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
}  // namespace

namespace optiling {
static ge::graphStatus All2AllDetourIoDieTiling(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto *tiling = context->GetTilingData<All2AllDetourIoDieTilingData>();
    const auto *sendShape = context->GetInputShape(0);
    const auto *commShape = context->GetInputShape(1);
    const auto *sendDesc = context->GetInputDesc(0);
    const auto *recvDesc = context->GetOutputDesc(0);
    OP_TILING_CHECK(tiling == nullptr || sendShape == nullptr || commShape == nullptr ||
                        sendDesc == nullptr || recvDesc == nullptr,
                    OP_LOGE(nodeName, "input/output metadata is null"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null"), return ge::GRAPH_FAILED);
    auto group = attrs->GetAttrPointer<char>(0);
    auto rankSize = attrs->GetAttrPointer<int64_t>(1);
    auto rankId = attrs->GetAttrPointer<int64_t>(2);
    auto magic = attrs->GetAttrPointer<int64_t>(3);
    OP_TILING_CHECK(group == nullptr || strnlen(group, MAX_GROUP_NAME_LENGTH) == 0UL ||
                        strnlen(group, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH,
                    OP_LOGE(nodeName, "group is invalid"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankSize == nullptr || *rankSize <= 2 || *rankSize > MAX_MTE_RANKS,
                    OP_LOGE(nodeName, "AIV+URMA detour requires rank_size in [3, %u]", MAX_MTE_RANKS),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankId == nullptr || *rankId < 0 || *rankId >= *rankSize,
                    OP_LOGE(nodeName, "rank_id is invalid"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(magic == nullptr || *magic <= 0,
                    OP_LOGE(nodeName, "magic must be positive"), return ge::GRAPH_FAILED);

    const uint64_t sendCount = NumElements(sendShape);
    const uint64_t commRankCount = NumElements(commShape);
    const uint64_t elementBytes = DataTypeBytes(sendDesc->GetDataType());
    OP_TILING_CHECK(elementBytes == 0UL || recvDesc->GetDataType() != sendDesc->GetDataType(),
                    OP_LOGE(nodeName, "unsupported or mismatched dtype"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commRankCount < 2UL || commRankCount >= static_cast<uint64_t>(*rankSize),
                    OP_LOGE(nodeName, "commRankIds must contain at least two ranks and leave one relay"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCount == 0UL || sendCount % commRankCount != 0UL,
                    OP_LOGE(nodeName, "sendData elements must be divisible by commRankIds size"),
                    return ge::GRAPH_FAILED);

    const uint64_t perRankBytes = sendCount / commRankCount * elementBytes;
    const uint64_t cellBytes = AlignUp(CELL_FLAG_BYTES + perRankBytes, CELL_ALIGN_BYTES);
    const uint64_t requiredWindow = A5_MTE_STATE_WIN_SIZE +
        static_cast<uint64_t>(*rankSize) * static_cast<uint64_t>(*rankSize) * cellBytes;
    const uint64_t availableWindow = Mc2TilingUtils::GetMaxWindowSize();
    OP_TILING_CHECK(availableWindow < requiredWindow,
                    OP_LOGE(nodeName, "HCCL window too small: need %lu bytes, got %lu",
                            requiredWindow, availableWindow), return ge::GRAPH_FAILED);

    tiling->info.rankSize = static_cast<uint32_t>(*rankSize);
    tiling->info.rankId = static_cast<uint32_t>(*rankId);
    tiling->info.commRankCount = static_cast<uint32_t>(commRankCount);
    tiling->info.magic = static_cast<uint32_t>(*magic);
    tiling->info.sendCount = sendCount;
    tiling->info.perRankBytes = perRankBytes;
    tiling->info.cellBytes = cellBytes;
    tiling->info.dataRegionOffset = A5_MTE_STATE_WIN_SIZE;

    constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
    AscendC::Mc2CcTilingConfig config(
        std::string(group), OP_TYPE_ALL_TO_ALL, "AlltoAll=level0:fullmesh;level1:pairwise");
    config.SetCommEngine(mc2tiling::AIV_ENGINE);
    config.GetTiling(tiling->mc2InitTiling);
    config.GetTiling(tiling->mc2CcTiling);

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE(nodeName, "workspace metadata is null"),
                    return ge::GRAPH_FAILED);
    workspace[0] = SYSTEM_WORKSPACE_BYTES;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    OP_TILING_CHECK(platform.GetCoreNumAiv() == 0U, OP_LOGE(nodeName, "no AIV core"),
                    return ge::GRAPH_FAILED);
    context->SetTilingKey(0UL);
    context->SetBlockDim(1U);
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

struct All2AllDetourIoDieCompileInfo {};
ge::graphStatus TilingParseForAll2AllDetourIoDie(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(All2AllDetourIoDie)
    .Tiling(All2AllDetourIoDieTiling)
    .TilingParse<All2AllDetourIoDieCompileInfo>(TilingParseForAll2AllDetourIoDie);
}  // namespace optiling
