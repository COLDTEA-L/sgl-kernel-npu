#!/usr/bin/env python3
"""Installed-wheel API/graph-shape regression, no NPU required."""
import ctypes
from pathlib import Path
import torch
import deep_ep.deep_ep_cpp as ext
from deep_ep import Buffer

library = ctypes.CDLL(str(Path(ext.__file__).resolve()))
fn = library.A5DeepEpPeerPlanAbiVersion
fn.restype = ctypes.c_int
assert fn() >= 2
assert hasattr(Buffer, "prepare_ccu_urma_peer_plan")
assert hasattr(Buffer, "bind_ccu_urma_peer_plan")
op = torch.ops.deep_ep.ccu_urma_peer_plan_alltoall
for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.int32):
    for k in (2, 4):
        x = torch.empty((k, 1024), device="meta", dtype=dtype)
        y = op(x, 1, [])
        assert y.shape == x.shape and y.dtype == x.dtype
        torch._dynamo.reset()
        captured = torch.compile(lambda t: op(t, 1, []), backend="eager", fullgraph=True)
        result = captured(x)
        assert result.shape == x.shape and result.dtype == dtype
for k in (1, 3, 8):
    try:
        torch.ops.deep_ep.ccu_urma_peer_plan_alltoall(torch.empty((k, 1024), device="meta"), 1, [])
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"accepted unsupported rank count {k}")
try:
    torch.ops.deep_ep.ccu_urma_peer_plan_alltoall(torch.empty((4, 1024), device="meta"), 1, [0])
except RuntimeError:
    pass
else:
    raise AssertionError("accepted zero weight")
print("installed peer-plan ABI, 2/4-rank Meta/fullgraph, invalid shape/weight: PASS")
