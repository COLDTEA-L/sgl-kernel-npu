#!/usr/bin/env python3
"""Compare Ascend 950 per-port HCCN counters for route0 and route2."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PREFERRED_TX = (
    "nic_tx_all_oct_num",
    "mac_tx_all_oct_num",
    "tx_all_oct_num",
    "tx_bytes",
)
PREFERRED_RX = (
    "nic_rx_all_oct_num",
    "mac_rx_all_oct_num",
    "rx_all_oct_num",
    "rx_bytes",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route0-tsv", required=True)
    parser.add_argument("--route2-tsv", required=True)
    parser.add_argument("--endpoint-devices", default="4,5")
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--tx-counter", default="")
    parser.add_argument("--rx-counter", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--min-signal-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()
    try:
        args.endpoints = {int(item) for item in args.endpoint_devices.split(",") if item}
    except ValueError:
        parser.error("--endpoint-devices must be comma-separated integers")
    if len(args.endpoints) != 2:
        parser.error("exactly two endpoint devices are required")
    if args.bytes <= 0 or args.iters <= 0 or args.min_signal_bytes < 0:
        parser.error("bytes/iters must be positive and min-signal-bytes non-negative")
    return args


def load_rows(path):
    rows = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append({
                "device": int(row["physical_device"]),
                "udie": row.get("udie", ""),
                "port": row.get("port", ""),
                "counter": row["counter"],
                "delta": int(row["delta"]),
            })
    return rows


def counter_score(name, direction):
    lowered = name.lower()
    if direction not in lowered or not ("oct" in lowered or "byte" in lowered):
        return -1
    if any(word in lowered for word in ("err", "drop", "retry", "rty", "pause")):
        return -1
    score = 0
    if "nic" in lowered:
        score += 40
    if "all" in lowered:
        score += 20
    if "oct" in lowered:
        score += 10
    return score


def choose_counter(names, configured, preferred, direction):
    if configured:
        if configured not in names:
            raise RuntimeError(f"counter {configured!r} not present; available={sorted(names)}")
        return configured
    for candidate in preferred:
        if candidate in names:
            return candidate
    candidates = sorted(
        ((counter_score(name, direction), name) for name in names), reverse=True
    )
    if not candidates or candidates[0][0] < 0:
        raise RuntimeError(
            f"cannot find a {direction.upper()} byte counter; available={sorted(names)}"
        )
    return candidates[0][1]


def aggregate(rows, tx_counter, rx_counter):
    result = defaultdict(lambda: {"tx": 0, "rx": 0})
    for row in rows:
        key = (row["device"], row["udie"], row["port"])
        if row["counter"] == tx_counter:
            result[key]["tx"] += max(row["delta"], 0)
        elif row["counter"] == rx_counter:
            result[key]["rx"] += max(row["delta"], 0)
    return result


def main():
    args = parse_args()
    route0_rows = load_rows(args.route0_tsv)
    route2_rows = load_rows(args.route2_tsv)
    names = {row["counter"] for row in route0_rows + route2_rows}
    tx_counter = choose_counter(names, args.tx_counter, PREFERRED_TX, "tx")
    rx_counter = choose_counter(names, args.rx_counter, PREFERRED_RX, "rx")
    route0 = aggregate(route0_rows, tx_counter, rx_counter)
    route2 = aggregate(route2_rows, tx_counter, rx_counter)
    keys = sorted(set(route0) | set(route2))
    expected = args.bytes * args.iters
    threshold = max(args.min_signal_bytes, expected // 100)

    endpoint_route0_max = max(
        (max(values["tx"], values["rx"])
         for key, values in route0.items() if key[0] in args.endpoints),
        default=0,
    )
    direct_threshold = max(threshold, endpoint_route0_max // 20)
    direct_ports = {
        key for key, values in route0.items()
        if key[0] in args.endpoints and max(values["tx"], values["rx"]) >= direct_threshold
    }
    direct_active = {
        key for key in direct_ports
        if max(route2[key]["tx"], route2[key]["rx"]) >= threshold
    }

    device_route0 = defaultdict(lambda: {"tx": 0, "rx": 0})
    device_route2 = defaultdict(lambda: {"tx": 0, "rx": 0})
    for key in keys:
        device = key[0]
        for direction in ("tx", "rx"):
            device_route0[device][direction] += route0[key][direction]
            device_route2[device][direction] += route2[key][direction]
    relay_candidates = []
    for device, values in sorted(device_route2.items()):
        if device in args.endpoints:
            continue
        baseline = device_route0[device]
        signal = min(values["tx"], values["rx"])
        baseline_signal = min(baseline["tx"], baseline["rx"])
        if signal >= threshold and signal >= max(threshold, baseline_signal * 4):
            relay_candidates.append(device)

    output = Path(args.output) if args.output else Path(args.route2_tsv).with_name(
        "hccn_route0_route2_comparison.tsv"
    )
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow((
            "physical_device", "udie", "port", "route0_tx", "route0_rx",
            "route2_tx", "route2_rx", "direct_port", "direct_active_in_route2",
            "relay_candidate_device",
        ))
        for key in keys:
            device, udie, port = key
            writer.writerow((
                device, udie, port, route0[key]["tx"], route0[key]["rx"],
                route2[key]["tx"], route2[key]["rx"],
                int(key in direct_ports), int(key in direct_active),
                int(device in relay_candidates),
            ))

    evidence = {
        "tx_counter": tx_counter,
        "rx_counter": rx_counter,
        "expected_bytes_per_direction": expected,
        "signal_threshold_bytes": threshold,
        "direct_ports": [list(key) for key in sorted(direct_ports)],
        "direct_ports_active_in_route2": [list(key) for key in sorted(direct_active)],
        "relay_candidate_devices": relay_candidates,
        "direct_plus_relay_evidence": bool(direct_active and relay_candidates),
        "comparison_tsv": str(output.resolve()),
    }
    print("HCCN_ROUTE_EVIDENCE " + json.dumps(evidence, sort_keys=True))
    print(f"TX counter: {tx_counter}; RX counter: {rx_counter}")
    print(f"Route0 direct ports: {sorted(direct_ports)}")
    print(f"Direct ports active in route2: {sorted(direct_active)}")
    print(f"Relay candidate devices: {relay_candidates}")
    print(f"Direct+relay cumulative evidence: {evidence['direct_plus_relay_evidence']}")
    print(f"Comparison table: {output.resolve()}")
    print("NOTE: before/after counters prove participation, not temporal concurrency.")


if __name__ == "__main__":
    main()
