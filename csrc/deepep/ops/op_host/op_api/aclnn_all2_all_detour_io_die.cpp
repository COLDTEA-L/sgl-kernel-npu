#include "aclnn_all2_all_detour_io_die.h"
#include "aclnnInner_all2_all_detour_io_die.h"

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
    aclnnStatus status = aclnnInnerAll2AllDetourIoDieGetWorkspaceSize(
        sendData, commRankIds, group, rankSize, rankId, recvData, workspaceSize, executor);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (executor == nullptr || *executor == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_CCU);
    }
    return status;
}

aclnnStatus aclnnAll2AllDetourIoDie(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnInnerAll2AllDetourIoDie(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
