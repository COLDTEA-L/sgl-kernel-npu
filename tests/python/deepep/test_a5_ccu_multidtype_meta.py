#!/usr/bin/env python3
"""Check compiled extension's dtype/Meta contracts without initializing NPU."""
import argparse
import ctypes
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu
    path = args.extension.resolve(strict=True)
    spec = importlib.util.spec_from_file_location("deep_ep_cpp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lib = ctypes.CDLL(str(path))
    version = lib.A5DeepEpPeerPlanAbiVersion
    version.restype = ctypes.c_int
    assert version() >= 2
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.int32):
        # Pattern is exactly representable even in BF16/INT8; no tolerance can
        # hide missing chunks or rank swaps in hardware correctness checks.
        from a5_ccu_peer_test_support import payload_code
        for epoch in (0, 2, 22, 42):
            values = torch.tensor([payload_code(i, 3, 2, epoch) for i in range(31)], dtype=torch.int32)
            assert torch.equal(values, values.to(dtype).to(torch.int32))
        for k in (2, 4):
            x = torch.empty((k, 1025), device="meta", dtype=dtype)
            y = torch.ops.deep_ep.ccu_urma_peer_plan_alltoall(x, 1, [])
            assert y.shape == x.shape and y.dtype == dtype
            # Each dtype/shape is an independent graph contract, not ten
            # specializations accumulated against Dynamo's default limit 8.
            torch._dynamo.reset()
            compiled = torch.compile(lambda t: torch.ops.deep_ep.ccu_urma_peer_plan_alltoall(t, 1, []),
                                     backend="eager", fullgraph=True)
            result = compiled(x)
            assert result.shape == x.shape and result.dtype == dtype
        x = torch.empty((2, 1025), device="meta", dtype=dtype)
        for op, extra in ((torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall, ()),
                          (torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall_policy, ([2, 1],))):
            y = op(x, 1, *extra)
            assert y.shape == x.shape and y.dtype == dtype
        print(f"Meta dtype PASS: {dtype}")
    for dtype in (torch.float64, torch.bool):
        try:
            torch.ops.deep_ep.ccu_urma_peer_plan_alltoall(torch.empty((2, 16), device="meta", dtype=dtype), 1, [])
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"unsupported dtype accepted: {dtype}")
    print("Multi-dtype Meta/fullgraph contracts PASS; not a hardware data/graph replay test")


if __name__ == "__main__":
    main()
