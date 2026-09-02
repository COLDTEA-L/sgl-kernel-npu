#ifndef ACLNN_ALL2ALL_DETOUR_IO_DIE_H_
#define ACLNN_ALL2ALL_DETOUR_IO_DIE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnAll2AllDetourIoDieGetWorkspaceSize(
    const aclTensor *sendData,
    const aclTensor *commRankIds,
    char *group,
    int64_t rankSize,
    int64_t rankId,
    int64_t magic,
    const aclTensor *recvData,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnAll2AllDetourIoDie(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
