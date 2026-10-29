"""Bitwise payload oracle; FP8 test data includes NaNs and signed zero."""
from ccu_payload import (PAYLOAD_DTYPE_NAMES, byte_payload_view, copy_payload_, fp8_dtypes)


def shifted_payload(tensor, epoch):
    import torch
    if tensor.dtype in fp8_dtypes():
        raw = byte_payload_view(tensor).to(torch.int16)
        return ((raw + epoch) % 256).to(torch.uint8).view(tensor.dtype)
    return tensor + epoch


def make_payload(rows, width, source, dtype, seed=0):
    import torch
    values = torch.arange(rows * width, dtype=torch.int64).reshape(rows, width)
    if dtype in fp8_dtypes():
        return ((values * 7 + source * 11 + seed) % 256).to(torch.uint8).view(dtype)
    return ((values * 7 + source * 11 + seed) % 97).to(dtype)


def cat_payload(pieces):
    import torch
    return torch.cat([byte_payload_view(t) for t in pieces], dim=0).view(pieces[0].dtype)


def full_payload(shape, sentinel, dtype, device):
    import torch
    if dtype in fp8_dtypes():
        return torch.full(shape, sentinel % 256, dtype=torch.uint8, device=device).view(dtype)
    return torch.full(shape, sentinel, dtype=dtype, device=device)


def fill_payload_(tensor, sentinel):
    byte_payload_view(tensor).fill_(sentinel % 256 if tensor.dtype in fp8_dtypes() else sentinel)


def equal_payload(left, right):
    import torch
    return left.dtype == right.dtype and torch.equal(byte_payload_view(left).cpu(), byte_payload_view(right).cpu())


def guards_intact(storage, guard, sentinel):
    raw = byte_payload_view(storage).cpu()
    if storage.dtype in fp8_dtypes(): sentinel %= 256
    return bool((raw[:guard] == sentinel).all() and (raw[-guard:] == sentinel).all())


def clone_payload(tensor):
    return byte_payload_view(tensor).clone().view(tensor.dtype)
