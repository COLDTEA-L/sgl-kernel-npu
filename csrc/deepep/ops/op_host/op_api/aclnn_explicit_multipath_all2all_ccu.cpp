#include "aclnn_explicit_multipath_all2all_ccu.h"
#include "aclnnInner_explicit_multipath_all2_all_ccu.h"

namespace {
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
}

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(
    void *executor, NnopbaseHcclServerType serverType);

extern "C" aclnnStatus aclnnExplicitMultipathAll2AllCcuGetWorkspaceSize(
    const aclTensor *sendData, char *group, int64_t rankSize, int64_t rankId,
    char *planId, char *pathWeights, const aclTensor *recvData,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus status = aclnnInnerExplicitMultipathAll2AllCcuGetWorkspaceSize(
        sendData, group, rankSize, rankId, planId, pathWeights,
        recvData, workspaceSize, executor);
    if (status == 0 && executor != nullptr && *executor != nullptr && NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_CCU);
    }
    return status;
}

extern "C" aclnnStatus aclnnExplicitMultipathAll2AllCcu(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnInnerExplicitMultipathAll2AllCcu(workspace, workspaceSize, executor, stream);
}
