/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef A5_CCU_URMA_ROUTE_PROBE_H
#define A5_CCU_URMA_ROUTE_PROBE_H

#include <acl/acl.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Public CCU SDK backend marker: 1 requires CANN public HcommCcu* APIs.
 * The original 9.1 branch does not export this capability. */
int A5CcuPublicApiAbiVersion(void);

HcclResult HcclCcuUrmaRouteProbe(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
                               aclrtStream stream);

/**
 * Two-rank CCU multi-route write used by the Python/PyTorch validation path.
 * A5_CCU_PATH_UIDS selects session-local discovered CommLink path objects.
 * A5_CCU_PATH_WEIGHTS controls peer-slice partitioning. Numeric route-index
 * variables remain available only for compatibility and diagnostics.
 * recvBuf contains rankSize consecutive sendCount-element source-rank slices.
 */
HcclResult HcclCcuUrmaMultiRouteWrite(void *sendBuf, void *recvBuf, uint64_t sendCount,
                                     HcclDataType dataType, HcclComm comm, aclrtStream stream);

/**
 * Two-rank CCU AllToAll over one or more RankGraph routes.
 * A5_CCU_SYNTHETIC_ROUTE_MANIFEST may instead provide multiple explicitly
 * resolved physical-relay EID pairs. Each manifest row becomes one Channel;
 * no relay card is selected implicitly in that mode.
 * sendBuf and recvBuf each contain two consecutive elementsPerPeer-element
 * slices. sendBuf[dstRank] is delivered to recvBuf[srcRank] on dstRank.
 */
HcclResult HcclCcuUrmaMultiRouteAllToAll(void *sendBuf, void *recvBuf,
                                        uint64_t elementsPerPeer, HcclDataType dataType,
                                        HcclComm comm, aclrtStream stream);

/**
 * Production-facing two-rank explicit multipath AllToAll.
 *
 * Channel 0 is the HCCL-discovered direct candidate selected by directRoute.
 * relayManifest contains one explicitly resolved relay EID pair per remaining
 * channel. pathWeights therefore contains relayCount + 1 entries in exactly
 * that order.  The normalized plan is passed as an ordinary API argument so a
 * future host controller can replace the command-line manifest producer
 * without changing the CCU kernel or relying on process-global environment
 * variables.  Prepared-plan ABI supports up to 64 total paths per launch;
 * the first path is direct and the remaining paths are explicit relays.
 */
HcclResult HcclCcuUrmaExplicitMultipathAllToAll(
    void *sendBuf, void *recvBuf, uint64_t elementsPerPeer,
    HcclDataType dataType, HcclComm comm, aclrtStream stream,
    const char *relayManifest, uint32_t directRoute,
    const uint32_t *pathWeights, uint32_t pathCount);

/** ABI marker for the prepared-plan API below.  ABI 5 adds explicit
 * stream binding and per-launch host path weights. */
int A5CcuUrmaPreparedPlanAbiVersion(void);
/** Overlap ABI 1: reusable Ex events, independent self-copy branch, explicit join.
 * Supported() checks an API symbol only, not hardware/capture functionality. */
int A5CcuCopyOverlapAbiVersion(void);
int A5CcuCopyOverlapSupported(void);

/** Independent peer-plan ABI: version 2 supports inferred FP16/BF16/FP32/INT8/INT32;
 * complete canonical peer manifest, 2/4 ranks,
 * direct-only or explicit communicator-external relay paths. */
int A5CcuPeerPlanAbiVersion(void);
// Channel metadata mask/submission protocol capability, not hardware route ID.
int A5CcuChannelSyncAbiVersion(void);
/** Destroy only outside capture, after synchronizing and releasing all graphs
 * using this handle. Reclaims owned self-copy streams/events; HCCL channel
 * cache lifetime remains owned by the existing communication runtime. */
HcclResult HcclCcuUrmaPeerPlanDestroy(uint64_t handle);
HcclResult HcclCcuUrmaExplicitMultipathPlanDestroy(uint64_t handle);
HcclResult HcclCcuUrmaPeerPlanCreate(HcclComm comm, aclrtStream stream,
    const char *planId, const char *manifest, uint32_t availableCards, uint64_t *handle);
HcclResult HcclCcuUrmaPeerPlanBindStream(uint64_t handle, HcclComm comm, aclrtStream stream);
HcclResult HcclCcuUrmaPeerPlanExecute(void *send, void *recv, uint64_t elementsPerPeer, uint32_t rankSize,
    HcclDataType dtype, HcclComm comm, aclrtStream stream, uint64_t handle,
    const uint32_t *policy, uint32_t policyCount);

/** Independent ABI 1: 8-rank direct-only row-count AllToAllV. All ranks pass
 * identical 64-entry counts[s*8+d]. Capacities are bytes and must be >=1 even
 * for an empty tensor. Out-of-place, baseline sync, one local IO die per rank.
 * BindStream and Destroy above also manage these handles. */
int A5CcuAllToAllVAbiVersion(void);
// Opt-in verification of discovered direct endpoint EIDs and local IO die.
int A5CcuBoundDirectAbiVersion(void);
HcclResult HcclCcuUrmaPeerPlanAllToAllVCreate(HcclComm comm, aclrtStream stream,
    const char *planId, const char *manifest, uint32_t availableCards, uint64_t *handle);
HcclResult HcclCcuUrmaPeerPlanAllToAllVExecute(void *send, void *recv,
    uint64_t sendCapacity, uint64_t recvCapacity, uint64_t rowBytes, uint32_t rank,
    HcclComm comm, aclrtStream stream, uint64_t handle, const uint64_t *counts, uint32_t count);

/**
 * Build and register all control-plane resources for one explicit multipath
 * plan. This call performs CommLink construction, HcclChannelAcquire and CCU
 * kernel registration. It must run once, outside the captured execution
 * graph, on the same ACL stream later used by PlanExecute.
 *
 * planId is an immutable, process-local controller key. Repeating PlanCreate
 * with the same communicator, stream, planId and configuration is idempotent.
 * Reusing the key with a different configuration is rejected.
 */
HcclResult HcclCcuUrmaExplicitMultipathPlanCreate(
    HcclComm comm, aclrtStream stream, const char *planId,
    const char *relayManifest, uint32_t directRoute,
    const uint32_t *pathWeights, uint32_t pathCount,
    uint64_t *planHandle);

/**
 * Bind an existing plan to an additional execution stream outside graph
 * capture.  Channel and CCU resources are stream-affine on T560, so graph
 * capture must call this before the first PlanExecute on that stream.
 */
HcclResult HcclCcuUrmaExplicitMultipathPlanBindStream(
    uint64_t planHandle, HcclComm comm, aclrtStream stream);

/**
 * Execute a previously prepared plan. This data-plane-only entry does not
 * parse a manifest, enumerate RankGraph links, acquire a Channel or register a
 * CCU kernel. It only updates buffer tokens/offsets and launches the cached
 * CCU kernel on the stream to which the plan was bound.
 */
HcclResult HcclCcuUrmaExplicitMultipathPlanExecute(
    void *sendBuf, void *recvBuf, uint64_t elementsPerPeer,
    HcclDataType dataType, HcclComm comm, aclrtStream stream,
    uint64_t planHandle);

/**
 * Execute with a per-launch host policy.  launchWeights has one positive
 * entry per prepared Channel; changing the values changes the byte split but
 * never rebuilds CommLinks, Channels or the CCU kernel.  This API is intended
 * for a host controller.  Graph replay keeps the values captured at graph
 * construction time.
 */
HcclResult HcclCcuUrmaExplicitMultipathPlanExecuteV2(
    void *sendBuf, void *recvBuf, uint64_t elementsPerPeer,
    HcclDataType dataType, HcclComm comm, aclrtStream stream,
    uint64_t planHandle, const uint32_t *launchWeights,
    uint32_t pathCount);

/**
 * Experimental cross-engine mailbox ABI.
 *
 * Create provisions explicit direct/relay Channels on a dedicated stream,
 * registers a persistent CCU worker and starts it polling commandBlock.  The
 * graph-visible AIV operator is the only producer of EXECUTE commands.
 */
int A5CcuHbmCommandPunctureAbiVersion(void);

HcclResult HcclCcuUrmaCommandBlockWorkerCreate(
    void *sendBuf, void *recvBuf, uint64_t elementsPerPeer,
    HcclDataType dataType, void *commandBlock, uint64_t commandBlockBytes,
    HcclComm comm, aclrtStream controlStream, const char *planId,
    const char *relayManifest, uint32_t directRoute,
    const uint32_t *pathWeights, uint32_t pathCount,
    uint64_t *workerHandle);

HcclResult HcclCcuUrmaCommandBlockWorkerStop(
    uint64_t workerHandle, aclrtStream controlStream);

// EP8 participant EID channels: no intermediate HBM payload, one route launch.
int A5CcuEidAllToAllVAbiVersion(void);
HcclResult HcclCcuUrmaPeerPlanSynchronize(uint64_t handle);
HcclResult HcclCcuUrmaEidAllToAllVCreate(HcclComm comm, aclrtStream stream,
    const char* planId, const char* manifest, uint64_t* handle);
// counts/relayRanks/relayRows are global row-major [8,8] host arrays, count=64.
// relayRanks are communicator ranks, -1 when relayRows=0; each flow uses at most one.
HcclResult HcclCcuUrmaEidAllToAllVExecute(void* send, void* recv,
    uint64_t sendCapacity, uint64_t recvCapacity, uint64_t rowBytes, uint32_t rank,
    HcclComm comm, aclrtStream stream, uint64_t handle, const uint64_t* counts,
    const int32_t* relayRanks, const uint64_t* relayRows, uint32_t count);

#ifdef __cplusplus
}
#endif

#endif // A5_CCU_URMA_ROUTE_PROBE_H
