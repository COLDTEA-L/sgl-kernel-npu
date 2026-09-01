#include <cstdint>
#include <cstring>
#include <string>

#include "error_log.h"
#include "mc2_tiling_utils.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "tiling/hccl/hccl_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/all2_all_detour_io_die_tiling.h"

namespace {
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
constexpr uint32_t MAX_CCU_RANKS = 32U;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr size_t SYSTEM_WORKSPACE_BYTES = 16UL * 1024UL * 1024UL;
constexpr int ATTR_GROUP = 0;
constexpr int ATTR_RANK_SIZE = 1;
constexpr int ATTR_RANK_ID = 2;

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
    if (type == ge::DT_FLOAT16 || type == ge::DT_BF16) {
        return 2UL;
    }
    if (type == ge::DT_FLOAT || type == ge::DT_INT32) {
        return 4UL;
    }
    return 0UL;
}
}  // namespace

namespace optiling {
static ge::graphStatus All2AllDetourIoDieTiling(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto *tiling = context->GetTilingData<All2AllDetourIoDieTilingData>();
    OP_TILING_CHECK(tiling == nullptr, OP_LOGE(nodeName, "tiling data is null"), return ge::GRAPH_FAILED);

    const auto *sendShape = context->GetInputShape(0);
    const auto *commShape = context->GetInputShape(1);
    const auto *sendDesc = context->GetInputDesc(0);
    const auto *recvDesc = context->GetOutputDesc(0);
    OP_TILING_CHECK(sendShape == nullptr || commShape == nullptr || sendDesc == nullptr || recvDesc == nullptr,
                    OP_LOGE(nodeName, "input/output metadata is null"), return ge::GRAPH_FAILED);

    const uint64_t sendCount = NumElements(sendShape);
    const uint64_t commRankCount = NumElements(commShape);
    const uint64_t elementBytes = DataTypeBytes(sendDesc->GetDataType());
    OP_TILING_CHECK(elementBytes == 0UL || recvDesc->GetDataType() != sendDesc->GetDataType(),
                    OP_LOGE(nodeName, "sendData and recvData must use the same supported dtype"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commRankCount == 0UL || commRankCount > MAX_CCU_RANKS,
                    OP_LOGE(nodeName, "commRankIds size must be in [1, %u]", MAX_CCU_RANKS),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCount == 0UL || sendCount % commRankCount != 0UL,
                    OP_LOGE(nodeName, "sendData element count must be divisible by commRankIds size"),
                    return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null"), return ge::GRAPH_FAILED);
    auto groupPtr = attrs->GetAttrPointer<char>(ATTR_GROUP);
    auto rankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_SIZE);
    auto rankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_ID);
    OP_TILING_CHECK(groupPtr == nullptr || strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == 0UL ||
                        strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH,
                    OP_LOGE(nodeName, "group is invalid"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankSizePtr == nullptr || rankIdPtr == nullptr,
                    OP_LOGE(nodeName, "rank attributes are null"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*rankSizePtr <= 0 || *rankSizePtr > static_cast<int64_t>(MAX_CCU_RANKS),
                    OP_LOGE(nodeName, "rank_size must be in [1, %u]", MAX_CCU_RANKS),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*rankIdPtr < 0 || *rankIdPtr >= *rankSizePtr,
                    OP_LOGE(nodeName, "rank_id must be in [0, rank_size)"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commRankCount > static_cast<uint64_t>(*rankSizePtr),
                    OP_LOGE(nodeName, "commRankIds size exceeds rank_size"), return ge::GRAPH_FAILED);

    tiling->info.rankSize = static_cast<uint32_t>(*rankSizePtr);
    tiling->info.rankId = static_cast<uint32_t>(*rankIdPtr);
    tiling->info.commRankCount = static_cast<uint32_t>(commRankCount);
    tiling->info.sendCount = sendCount;
    tiling->info.perRankBytes = (sendCount / commRankCount) * elementBytes;

    AscendC::Mc2CcTilingConfig ccuConfig(
        std::string(groupPtr), OP_TYPE_ALL_TO_ALL, "AlltoAll=level0:fullmesh;level1:pairwise");
    ccuConfig.SetCommEngine(mc2tiling::AIV_ENGINE);
    ccuConfig.GetTiling(tiling->mc2InitTiling);
    ccuConfig.GetTiling(tiling->mc2CcTiling);

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE(nodeName, "workspace metadata is null"),
                    return ge::GRAPH_FAILED);
    workspace[0] = SYSTEM_WORKSPACE_BYTES;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aivNum = platform.GetCoreNumAiv();
    OP_TILING_CHECK(aivNum < 2U, OP_LOGE(nodeName, "CCU AlltoAllV requires at least two AIV cores"),
                    return ge::GRAPH_FAILED);
    context->SetTilingKey(1UL);
    context->SetBlockDim(platform.CalcTschBlockDim(aivNum, 0U, aivNum));
    context->SetScheduleMode(1);
    OP_LOGI(nodeName,
            "A5 IO Die All2AllV: rank=%ld/%ld commRanks=%lu sendCount=%lu perRankBytes=%lu aivNum=%u",
            *rankIdPtr, *rankSizePtr, commRankCount, sendCount, tiling->info.perRankBytes, aivNum);
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
