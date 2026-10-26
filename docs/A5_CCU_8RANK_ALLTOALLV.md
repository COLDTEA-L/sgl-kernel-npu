# EP8 direct-only CCU AllToAllV

This is a separate prepared runtime API. Existing 2/4-rank equal-split APIs,
their argument packing, kernels and ABI markers are unchanged.

## Contract

- Exactly eight ranks, one direct Channel per remote peer; no relay support.
- Each rank's seven Channels must use one local IO die. Mixed local dies fail
  during preparation; this version does not implement a multi-die scheduler.
- Baseline metadata protocol only. Set `A5_CCU_CHANNEL_SYNC_MODE=baseline`.
- Contiguous tensors `[rows, ...]` with equal, nonzero trailing dimensions and
  matching FP16/BF16/FP32/INT8/INT32 dtype. FP8/MXFP are not supported.
- Every rank supplies the **same** 64 host integers in source-major order:
  `counts[source * 8 + destination]`, including self. These are row counts,
  not bytes, and need not be symmetric. Consistency across processes is the
  caller's collective contract; there is no extra count-consistency collective.
- Send rows are grouped by destination; receive rows are grouped by source.
  Shapes must exactly match the local row sum / column sum. Buffers are
  out-of-place. All-zero collectives, empty senders, empty receivers and empty
  self rows are legal. The bridge supplies registered token backing for empty
  tensors without sending dummy rows.
- Source/target offsets and lengths use checked 64-bit byte arithmetic. Self
  source and destination offsets are independent. There is no per-peer padding.
- Zero remote lengths skip both WriteNb and its event wait. Every Channel
  still publishes address/token and completes its baseline end handshake.
- Self-copy is the existing 8 x 32 KiB GroupCopy. Its zero control tuple skips
  the copy. A kernel launch has 10 fixed + 7 x 3 path parameters = 31.
- Preparation and stream binding are outside execution. Use sequential eager
  calls initially. Graph, concurrent stream use, physical path and NPU
  correctness/performance are separate validations, not implied by CPU/Meta.

## Prepare and execute

Generate the 28 direct peer entries, preserving the intended logical-to-physical
mapping (replace the ID order with the actual communication group mapping):

```bash
python3 scripts/prepare_a5_ccu_peer_plan.py \
  --devices 0,1,2,3,4,5,6,7 --available-phys 0,1,2,3,4,5,6,7 \
  --direct-route 0 --output-dir /tmp/ep8-direct-plan
```

`direct-route` is a candidate ordinal. Check it on the target machine; 0 is not
a universal physical route ID. Preparation verifies that the selected link is
direct, checks endpoint device identity and enforces a common local IO die.

```python
# EP group and deep_ep.Buffer already initialized on the correct device.
handle = buffer.prepare_ccu_urma_peer_plan_alltoallv(
    "ep8-direct-v1", "/tmp/ep8-direct-plan/peer_plan.tsv")
# To use another execution stream, bind it while that stream is current:
# buffer.bind_ccu_urma_peer_plan_alltoallv(handle)
buffer.ccu_urma_peer_plan_alltoallv_out(send, recv, handle, counts)
```

There is also an out torch operator (rank must match the plan owner):

```python
torch.ops.deep_ep.ccu_urma_peer_plan_alltoallv_out(
    send, recv, handle, rank, counts)
```

Synchronize outstanding uses before calling the existing C API
`HcclCcuUrmaPeerPlanDestroy(handle)` or tearing down the communicator.
Do not resize/release input or output storage while the launch is in flight.
The route library exports `A5CcuAllToAllVAbiVersion() == 1`; the bridge exports
`A5DeepEpAllToAllVAbiVersion() == 1`. Rebuild both components. Neither marker
changes the system HCCL/CANN version.

## Checks

```bash
python3 -m unittest discover -s tests/python/deepep -p 'test_a5_ccu_alltoallv_cpu.py' -v
```

The test compiles the **production** AllToAllV kernel against a CPU instruction
recorder, then runs an eight-rank notification scheduler for 100 changing
collectives (800 rank launches). It checks asymmetric rows, independent self
offsets, zero sends/receives/self, completely empty collectives, byte tails,
guard regions, immutable inputs, skipped events, drained notifications, and
overflow/invalid layout rejection. GroupCopy/MS and CCU hardware are modeled.

On an eight-NPU test machine, after installing matching route/bridge builds,
run the separate worker with the usual environment/runtime bootstrap:

```bash
torchrun --nproc_per_node=8 tests/python/deepep/test_a5_ccu_alltoallv_npu.py \
  --manifest /tmp/ep8-direct-plan/peer_plan.tsv --run-dir /tmp/ep8-check-unique
```

Use a fresh run directory. All rank JSON results and successful process exit
are required. The worker compares against native HCCL AllToAllV as well as an
independent token reference, changes payloads, poisons output and checks guards.
This is correctness-only, not a benchmark or graph test.

## Development validation (2026-10-26)

- Development container: `cam_lyw_dev_91`, CANN `9.1.0-beta.1`, conda
  `cam_py311_pt28` (Python 3.11, torch 2.8 / torch_npu 2.8).
- CPU suite: `test_a5_ccu_*cpu.py`, 67 tests passed, including the new production
  graph scheduler and prior 2/4-rank regressions.
- Route custom library compiled and `.run` packaged in build-only mode; new
  AllToAllV exports and existing prepared 16 / peer 5 / sync 1 ABI checked.
- `cmake --build build --target deep_ep_cpp -j4` passed. The built extension is
  `build/csrc/deepep/deep_ep_cpp.cpython-311-aarch64-linux-gnu.so`.
- Compiled bridge Meta gate: 160 valid EP8 cases, 12 invalid inputs rejected,
  and 10 legacy 2/4-rank cases passed. This is a shape/API check only.
- No system library/package installation, no eight-NPU run, no graph test,
  no physical-path or performance result. Compilation does not validate a
  different SDK environment.
