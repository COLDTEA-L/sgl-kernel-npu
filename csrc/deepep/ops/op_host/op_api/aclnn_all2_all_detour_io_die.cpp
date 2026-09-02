#include "aclnn_all2_all_detour_io_die.h"
#include "aclnnInner_all2_all_detour_io_die.h"

#include <cstdio>
#include <cstdlib>

#ifndef ACLNN_ERR_INNER_NULLPTR
#define ACLNN_ERR_INNER_NULLPTR (-1)
#endif

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif

namespace {
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

bool DetourTraceEnabled()
{
    const char *value = std::getenv("A5_DETOUR_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void TraceDetour(const char *stage)
{
    if (!DetourTraceEnabled()) {
        return;
    }
    const char *rank = std::getenv("RANK");
    std::fprintf(stderr, "[A5 detour][rank=%s] %s\n", rank == nullptr ? "?" : rank, stage);
    std::fflush(stderr);
}
}  // namespace

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(
    void *executor, NnopbaseHcclServerType serverType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnAll2AllDetourIoDieGetWorkspaceSize(
    const aclTensor *sendData,
    const aclTensor *commRankIds,
    char *group,
    int64_t rankSize,
    int64_t rankId,
    const aclTensor *recvData,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    TraceDetour("GetWorkspaceSize begin");
    aclnnStatus status = aclnnInnerAll2AllDetourIoDieGetWorkspaceSize(
        sendData, commRankIds, group, rankSize, rankId, recvData, workspaceSize, executor);
    if (DetourTraceEnabled()) {
        const char *rank = std::getenv("RANK");
        const unsigned long long bytes =
            workspaceSize == nullptr ? 0ULL : static_cast<unsigned long long>(*workspaceSize);
        std::fprintf(stderr, "[A5 detour][rank=%s] GetWorkspaceSize end: status=%d, workspace=%llu\n",
                     rank == nullptr ? "?" : rank, static_cast<int>(status), bytes);
        std::fflush(stderr);
    }
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (executor == nullptr || *executor == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_CCU);
        TraceDetour("HCCL server type set to CCU");
    } else {
        TraceDetour("NnopbaseSetHcclServerType symbol is unavailable");
    }
    return status;
}

aclnnStatus aclnnAll2AllDetourIoDie(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    TraceDetour("execute begin");
    aclnnStatus status = aclnnInnerAll2AllDetourIoDie(workspace, workspaceSize, executor, stream);
    if (DetourTraceEnabled()) {
        const char *rank = std::getenv("RANK");
        std::fprintf(stderr, "[A5 detour][rank=%s] execute end: status=%d\n",
                     rank == nullptr ? "?" : rank, static_cast<int>(status));
        std::fflush(stderr);
    }
    return status;
}

#ifdef __cplusplus
}
#endif
