#!/usr/bin/env python3
"""Exercise the compiled bridge on Meta tensors. No NPU execution is implied."""
import argparse
import ctypes
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    sys.path.insert(0, str(args.extension.resolve().parent))
    import deep_ep_cpp
    if Path(deep_ep_cpp.__file__).resolve() != args.extension.resolve():
        raise RuntimeError("loaded a different DeepEP extension")
    library = ctypes.CDLL(str(args.extension.resolve()))
    marker = library.A5DeepEpAllToAllVAbiVersion
    marker.restype = ctypes.c_int
    assert marker() == 1
    op = torch.ops.deep_ep.ccu_urma_peer_plan_alltoallv_out
    cases = [
        [[0] * 8 for _ in range(8)],
        [[s + 1 if s == d else 0 for d in range(8)] for s in range(8)],
        [[(s * 7 + d * 3) % 11 for d in range(8)] for s in range(8)],
        [[13 if s == 2 and d == 5 else 0 for d in range(8)] for s in range(8)],
    ]
    checked = 0
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.int32):
        for matrix in cases:
            counts = sum(matrix, [])
            for rank in range(8):
                send = torch.empty((sum(matrix[rank]), 257), dtype=dtype, device="meta")
                recv = torch.empty((sum(row[rank] for row in matrix), 257), dtype=dtype, device="meta")
                result = op(send, recv, 1, rank, counts)
                assert result.shape == recv.shape and result.dtype == dtype
                checked += 1
        # Existing equal-split interfaces retain both 2/4-rank shape contracts.
        for ranks in (2, 4):
            send = torch.empty((ranks, 17), dtype=dtype, device="meta")
            assert torch.ops.deep_ep.ccu_urma_peer_plan_alltoall(send, 1, []).shape == send.shape
    empty = torch.empty((0, 3), device="meta")
    zeros = [0] * 64
    bad = [
        (empty, empty, 0, 0, zeros),
        (empty, empty, 1, 8, zeros),
        (empty, empty, 1, -1, zeros),
        (empty, empty, 1, 0, zeros[:-1]),
        (empty, empty, 1, 0, [-1] + zeros[1:]),
        (torch.empty((1, 3), device="meta"), empty, 1, 0, zeros),
        (empty, torch.empty((1, 3), device="meta"), 1, 0, zeros),
        (empty, torch.empty((0, 4), device="meta"), 1, 0, zeros),
        (empty, empty.to(torch.int32), 1, 0, zeros),
        (empty.to(torch.int64), empty.to(torch.int64), 1, 0, zeros),
        (torch.empty((0, 0), device="meta"), torch.empty((0, 0), device="meta"), 1, 0, zeros),
        (empty, empty, 1, 0, [2**63 - 1] + zeros[1:]),
    ]
    for arguments in bad:
        try:
            op(*arguments)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid AllToAllV arguments accepted")
    print(f"ALLTOALLV_META_PASS valid={checked} invalid={len(bad)} legacy=10")


if __name__ == "__main__":
    main()
