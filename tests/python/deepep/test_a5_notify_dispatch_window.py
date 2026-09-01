#!/usr/bin/env python3
import argparse
import os

import deep_ep
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="A5 NotifyDispatch communication-window smoke test")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--topk", type=int, default=2)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("Use at least two A5 NPUs to validate a remote communication-window address")

    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    group = dist.new_group(list(range(world_size)))

    num_experts = world_size * args.experts_per_rank
    values = torch.arange(args.tokens * args.topk, dtype=torch.int64, device="npu")
    topk_idx = ((values + rank) % num_experts).reshape(args.tokens, args.topk)
    topk_weights = torch.ones_like(topk_idx, dtype=torch.float32)
    expert_rank = topk_idx // args.experts_per_rank
    num_tokens_per_rank = torch.stack(
        [(expert_rank == r).any(dim=1).sum() for r in range(world_size)]
    ).to(torch.int32)
    num_tokens_per_expert = torch.stack(
        [(topk_idx == expert).sum() for expert in range(num_experts)]
    ).to(torch.int32)
    is_token_in_rank = torch.stack(
        [(expert_rank == r).any(dim=1) for r in range(world_size)], dim=1
    ).to(torch.int32)
    x = torch.zeros((args.tokens, 16), dtype=torch.bfloat16, device="npu")

    buffer = deep_ep.Buffer(group, num_nvl_bytes=0, num_rdma_bytes=0)
    outputs = buffer.notify_verify(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
    )
    torch.npu.synchronize()

    total_recv_tokens = outputs[6]
    recv_tokens_per_expert = outputs[8]
    expected_expert_counts = num_tokens_per_expert.clone()
    dist.all_reduce(expected_expert_counts, group=group)
    begin = rank * args.experts_per_rank
    end = begin + args.experts_per_rank
    expected_local_counts = expected_expert_counts[begin:end]
    torch.testing.assert_close(recv_tokens_per_expert.flatten(), expected_local_counts)
    expected_total = expected_local_counts.sum().to(torch.int32).reshape(1)
    torch.testing.assert_close(total_recv_tokens.flatten(), expected_total)

    if rank == 0:
        print("PASS: A5 NotifyDispatch acquired and used the HCCL communication window")
        print(f"world_size={world_size}, tokens={args.tokens}, experts={num_experts}, topk={args.topk}")

    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
