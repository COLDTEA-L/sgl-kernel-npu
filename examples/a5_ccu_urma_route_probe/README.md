# A5 CCU + URMA route probe

This custom HCCL operation is an AllGather-shaped two-rank probe. It submits
`HcommWriteOnThread` on a `COMM_ENGINE_CCU` thread using the local and remote
HCCL CCL buffers, enumerates every
`COMM_PROTOCOL_UBC_CTP` link exposed
by the HCCL RankGraph, prints its endpoint EIDs and hop count, and selects one
link with `A5_CCU_ROUTE_INDEX`.

The probe is the first validation stage for explicit IO Die routing. It proves
that a selected RankGraph route can carry CCU traffic correctly. It does not
invent a route that is absent from RankGraph and is not yet the final detour
AllToAll implementation.

The testcase performs each iteration in two phases: both ranks enqueue and
synchronize the remote write, meet at a host barrier, then copy the received
CCL-buffer slice into the output buffer and meet at a second barrier.

See `docs/A5_CCU_URMA_ROUTE_VALIDATION.md` in the sgl-kernel-npu repository for
build, install, run, profiling, and pass/fail instructions.
