# A5 CCU + URMA route probe

This custom HCCL operation is an AllGather-shaped two-rank probe. It enumerates
the `COMM_PROTOCOL_UBC_CTP` links exposed by the HCCL RankGraph, prints their
endpoint addresses and hop counts, and selects one link with
`A5_CCU_ROUTE_INDEX`. The selected channel is compiled into a registered CCU
kernel; each invocation launches that kernel with `HcclCcuKernelLaunch`, and
the kernel performs the peer transfer with CCU `WriteNb`.

The probe is the first validation stage for explicit IO Die routing. It proves
that a selected RankGraph route can carry CCU traffic correctly. It does not
invent a route that is absent from RankGraph and is not yet the final detour
AllToAll implementation.

The probe passes the user-buffer addresses and CCU tokens directly to the
kernel. Peer notifications in the kernel establish the cross-rank ordering;
there is no host `HcommWriteOnThread`, CCL-buffer staging, readback phase, or
host barrier.

See `docs/A5_CCU_URMA_ROUTE_VALIDATION.md` in the sgl-kernel-npu repository for
build, install, run, profiling, and pass/fail instructions.
