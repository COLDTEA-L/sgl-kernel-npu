#!/usr/bin/env python3
"""Aggregate isolated CCU AllToAll benchmark process results."""

import argparse
import json
import statistics


def values(items):
    return [float(json.loads(item)["host_batch_avg_us"]) for item in items]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--route0-bytes", type=int, required=True)
    parser.add_argument("--route2-bytes", type=int, required=True)
    parser.add_argument("--native", nargs="+", required=True)
    parser.add_argument("--route0", nargs="+", required=True)
    parser.add_argument("--route2", nargs="+", required=True)
    parser.add_argument("--serial", nargs="+", required=True)
    parser.add_argument("--concurrent", nargs="+", required=True)
    parser.add_argument("--require-overlap-ratio", type=float)
    args = parser.parse_args()

    native_us = statistics.median(values(args.native))
    route0_us = statistics.median(values(args.route0))
    route2_us = statistics.median(values(args.route2))
    serial_us = statistics.median(values(args.serial))
    concurrent_us = statistics.median(values(args.concurrent))
    speedup = serial_us / concurrent_us
    overlap_ratio = (serial_us - concurrent_us) / min(route0_us, route2_us)
    if overlap_ratio >= 0.50:
        evidence = "strong"
    elif overlap_ratio >= 0.15:
        evidence = "partial"
    elif overlap_ratio <= 0.05:
        evidence = "none"
    else:
        evidence = "inconclusive"

    result = {
        "total_bytes": args.bytes,
        "route0_bytes": args.route0_bytes,
        "route2_bytes": args.route2_bytes,
        "native_hccl_ccu_us": native_us,
        "route0_share_us": route0_us,
        "route2_share_us": route2_us,
        "multiroute_serial_us": serial_us,
        "multiroute_concurrent_us": concurrent_us,
        "multiroute_vs_native": native_us / concurrent_us,
        "serial_to_concurrent_speedup": speedup,
        "overlap_ratio": overlap_ratio,
        "evidence": evidence,
    }
    print("CONCURRENCY_RESULT " + json.dumps(result, sort_keys=True))
    if args.require_overlap_ratio is not None and overlap_ratio < args.require_overlap_ratio:
        raise SystemExit(
            f"overlap_ratio {overlap_ratio:.4f} is below required "
            f"{args.require_overlap_ratio:.4f}")


if __name__ == "__main__":
    main()
