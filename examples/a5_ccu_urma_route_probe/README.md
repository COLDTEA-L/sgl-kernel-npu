# A5 CCU + URMA route probe

This custom HCCL operation is an AllGather-shaped two-rank probe. It enumerates
the `COMM_PROTOCOL_UBC_CTP` links exposed by the HCCL RankGraph, prints their
endpoint addresses and hop counts, and selects one link with
`A5_CCU_ROUTE_INDEX`. The selected channel is compiled into a registered CCU
kernel; each invocation launches that kernel with `HcclCcuKernelLaunch`, and
the kernel performs the peer transfer with CCU `WriteNb`.

The original probe is the first validation stage for explicit IO Die routing.
The same package now also exports `HcclCcuUrmaMultiRouteAllToAll`, a true
two-rank AllToAll. Each rank provides one self slice and one peer slice; the
peer slice can use route 0, route 2, or route 0+2 with a default 2:1 split.
Neither interface invents a route that is absent from RankGraph.

The probe passes the user-buffer addresses and CCU tokens directly to the
kernel. Peer notifications in the kernel establish the cross-rank ordering;
there is no host `HcommWriteOnThread`, CCL-buffer staging, readback phase, or
host barrier.

The AllToAll kernel follows native HCCL ordering: exchange remote output/token,
enqueue CCU `LocalCopyNb` and all `WriteNb` operations, wait for their completion,
then perform one peer-level completion handshake. Per-launch logging is disabled
unless `A5_CCU_DEBUG=1` is set.

See `docs/A5_CCU_URMA_ROUTE_VALIDATION.md` in the sgl-kernel-npu repository for
build, install, run, profiling, and pass/fail instructions.
