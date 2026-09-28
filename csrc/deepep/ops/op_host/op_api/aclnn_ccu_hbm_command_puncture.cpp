#include "aclnn_ccu_hbm_command_puncture.h"
#include "aclnnInner_ccu_hbm_command_puncture.h"

extern "C" aclnnStatus aclnnCcuHbmCommandPunctureGetWorkspaceSize(
    const aclTensor *sendData, const aclTensor *recvData,
    const aclTensor *commandBlock, int64_t rankId, char *pathWeights,
    bool transfer, const aclTensor *ack, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerCcuHbmCommandPunctureGetWorkspaceSize(
        sendData, recvData, commandBlock, rankId, pathWeights, transfer,
        ack, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnCcuHbmCommandPuncture(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    return aclnnInnerCcuHbmCommandPuncture(
        workspace, workspaceSize, executor, stream);
}
