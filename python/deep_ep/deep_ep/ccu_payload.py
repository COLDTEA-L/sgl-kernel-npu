"""Dtypes for byte-preserving CCU transport. Import-safe without Torch."""
from functools import lru_cache

FP8_DTYPE_NAMES = ('float8_e4m3fn', 'float8_e5m2',
                   'float8_e4m3fnuz', 'float8_e5m2fnuz')
PAYLOAD_DTYPE_NAMES = ('float16', 'bfloat16', 'float32', 'int8', 'int32') + FP8_DTYPE_NAMES


@lru_cache(None)
def fp8_dtypes():
    import torch
    return tuple(getattr(torch, name) for name in FP8_DTYPE_NAMES if hasattr(torch, name))


@lru_cache(None)
def payload_dtypes():
    import torch
    return tuple(getattr(torch, name) for name in PAYLOAD_DTYPE_NAMES if hasattr(torch, name))


def byte_payload_view(tensor):
    """No cast: native HCCL and copies can transport every FP8 bit pattern."""
    import torch
    return tensor.view(torch.uint8) if tensor.dtype in fp8_dtypes() else tensor


def copy_payload_(dst, src):
    if dst.dtype != src.dtype:
        raise ValueError('CCU payload copy requires matching dtypes')
    byte_payload_view(dst).copy_(byte_payload_view(src))
    return dst
