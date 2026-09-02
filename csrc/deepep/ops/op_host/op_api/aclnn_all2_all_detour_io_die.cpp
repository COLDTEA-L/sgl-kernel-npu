#include "aclnn_all2_all_detour_io_die.h"
#include "aclnnInner_all2_all_detour_io_die.h"

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
    const aclTensor *sendData, const aclTensor *commRankIds, char *group,
    int64_t rankSize, int64_t rankId, int64_t magic, const aclTensor *recvData,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerAll2AllDetourIoDieGetWorkspaceSize(
        sendData, commRankIds, group, rankSize, rankId, magic, recvData, workspaceSize, executor);
}

aclnnStatus aclnnAll2AllDetourIoDie(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    // MTE creates the A5 windows used as URMA-mapped remote GM.  The custom
    // AIV kernel performs both one-sided phases; no HCCL collective is queued.
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerAll2AllDetourIoDie(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
