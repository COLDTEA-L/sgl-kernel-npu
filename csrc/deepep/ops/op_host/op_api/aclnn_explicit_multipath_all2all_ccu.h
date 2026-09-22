#ifndef ACLNN_EXPLICIT_MULTIPATH_ALL2ALL_CCU_H_
#define ACLNN_EXPLICIT_MULTIPATH_ALL2ALL_CCU_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnExplicitMultipathAll2AllCcuGetWorkspaceSize(
    const aclTensor *sendData, char *group, int64_t rankSize, int64_t rankId,
    char *planId, char *pathWeights, const aclTensor *recvData,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnExplicitMultipathAll2AllCcu(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
