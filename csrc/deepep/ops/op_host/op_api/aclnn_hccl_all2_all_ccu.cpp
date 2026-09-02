#include "aclnn_hccl_all2_all_ccu.h"
#include "aclnnInner_hccl_all2_all_ccu.h"

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

aclnnStatus aclnnHcclAll2AllCcuGetWorkspaceSize(
    const aclTensor *sendData, char *group, int64_t rankSize, int64_t rankId,
    const aclTensor *recvData, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus status = aclnnInnerHcclAll2AllCcuGetWorkspaceSize(
        sendData, group, rankSize, rankId, recvData, workspaceSize, executor);
    if (status == 0 && executor != nullptr && *executor != nullptr && NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_CCU);
    }
    return status;
}

aclnnStatus aclnnHcclAll2AllCcu(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnInnerHcclAll2AllCcu(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
