# A5 CCU + URMA route probe

This custom HCCL operation is an AllGather-shaped two-rank probe. It submits
HCOMM primitives on a `COMM_ENGINE_CCU` thread, enumerates every
`COMM_PROTOCOL_UBC_CTP` link exposed
by the HCCL RankGraph, prints its endpoint EIDs and hop count, and selects one
link with `A5_CCU_ROUTE_INDEX`.

The probe is the first validation stage for explicit IO Die routing. It proves
that a selected RankGraph route can carry CCU traffic correctly. It does not
invent a route that is absent from RankGraph and is not yet the final detour
AllToAll implementation.

See `docs/A5_CCU_URMA_ROUTE_VALIDATION.md` in the sgl-kernel-npu repository for
build, install, run, profiling, and pass/fail instructions.
