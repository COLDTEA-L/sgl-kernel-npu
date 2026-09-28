#ifndef ACLNN_CCU_HBM_COMMAND_PUNCTURE_H_
#define ACLNN_CCU_HBM_COMMAND_PUNCTURE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnCcuHbmCommandPunctureGetWorkspaceSize(
    const aclTensor *sendData, const aclTensor *recvData,
    const aclTensor *commandBlock, int64_t rankId, char *pathWeights,
    bool transfer, const aclTensor *ack, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnCcuHbmCommandPuncture(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
