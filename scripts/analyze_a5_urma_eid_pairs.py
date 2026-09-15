#!/usr/bin/env python3
"""Convert A5 HCCN snapshots into a stable EID-pair-to-relay map."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


PREFERRED_TX = ("nic_tx_all_oct_num", "tx_busi_flit_num", "ub_mem_pkt_cnt_tx")
PREFERRED_RX = ("nic_rx_all_oct_num", "rx_busi_flit_num", "ub_mem_pkt_cnt_rx")
COUNTER_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*(?::|=|\s)\s*(0[xX][0-9A-Fa-f]+|[0-9]+)\s*$"
)
DEVICE_RE = re.compile(
    r"^(before|after)_device([0-9]+)(?:_udie([0-9]+)_port([0-9]+))?\.txt$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot-dir")
    mode.add_argument("--scan-dir")
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--endpoint-devices", default="6,7")
    parser.add_argument("--tx-counter", default="")
    parser.add_argument("--rx-counter", default="")
    parser.add_argument("--min-count", type=int, default=0)
    parser.add_argument("--min-ratio", type=float, default=0.01)
    parser.add_argument("--dominance-ratio", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.endpoints = {int(item) for item in args.endpoint_devices.split(",") if item}
    if args.scan_dir and len(args.endpoints) != 2:
        parser.error("--endpoint-devices requires exactly two IDs")
    if args.min_count < 0 or not 0 <= args.min_ratio <= 1 or args.dominance_ratio < 1:
        parser.error("invalid classification threshold")
    return args


def read_snapshot(path: Path):
    counters = defaultdict(int)
    for line in path.read_text(errors="replace").splitlines():
        match = COUNTER_RE.match(line)
        if match:
            counters[match.group(1)] += int(match.group(2), 0)
    return counters


def snapshot_delta(snapshot_dir: Path, output: Path):
    files = defaultdict(dict)
    for path in snapshot_dir.glob("*_device*.txt"):
        match = DEVICE_RE.match(path.name)
        if match:
            key = (int(match.group(2)), match.group(3) or "", match.group(4) or "")
            files[key][match.group(1)] = path
    rows = []
    for (device, udie, port), phases in sorted(files.items()):
        if "before" not in phases or "after" not in phases:
            continue
        before = read_snapshot(phases["before"])
        after = read_snapshot(phases["after"])
        for counter in sorted(before.keys() & after.keys()):
            delta = after[counter] - before[counter]
            rows.append((device, udie, port, counter, before[counter], after[counter], max(delta, 0)))
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("physical_device", "udie", "port", "counter", "before", "after", "delta"))
        writer.writerows(rows)
    if not rows:
        raise RuntimeError(f"no matching HCCN counters in {snapshot_dir}")
    return rows


def load_delta(path: Path):
    with path.open(newline="") as handle:
        return [
            {
                "device": int(row["physical_device"]),
                "counter": row["counter"],
                "delta": int(row["delta"]),
            }
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def counter_score(name, direction):
    lowered = name.lower()
    if direction not in lowered or any(token in lowered for token in ("err", "retry", "ack", "drop")):
        return -1
    score = 0
    if "oct" in lowered or "byte" in lowered:
        score += 50
    if "busi_flit" in lowered:
        score += 40
    if "mem_pkt" in lowered:
        score += 20
    return score


def choose_counter(names, requested, preferred, direction):
    if requested:
        if requested not in names:
            raise RuntimeError(f"counter {requested!r} is unavailable")
        return requested
    for name in preferred:
        if name in names:
            return name
    ranked = sorted(((counter_score(name, direction), name) for name in names), reverse=True)
    if not ranked or ranked[0][0] < 0:
        raise RuntimeError(f"cannot identify a {direction} traffic counter; available={sorted(names)}")
    return ranked[0][1]


def classify(rows, endpoints, tx_counter, rx_counter, min_count, min_ratio, dominance_ratio):
    traffic = defaultdict(lambda: {"tx": 0, "rx": 0})
    for row in rows:
        if row["counter"] == tx_counter:
            traffic[row["device"]]["tx"] += row["delta"]
        elif row["counter"] == rx_counter:
            traffic[row["device"]]["rx"] += row["delta"]
    endpoint_signal = max(
        (max(value["tx"], value["rx"]) for dev, value in traffic.items() if dev in endpoints),
        default=0,
    )
    threshold = max(min_count, int(endpoint_signal * min_ratio), 1)
    scores = {
        dev: min(value["tx"], value["rx"])
        for dev, value in traffic.items()
        if dev not in endpoints
    }
    candidates = sorted(
        ((score, dev) for dev, score in scores.items() if score >= threshold), reverse=True
    )
    if not candidates:
        result = "DIRECT_ONLY_OR_NO_RELAY_SIGNAL"
        relay = ""
    elif len(candidates) == 1 or candidates[0][0] >= candidates[1][0] * dominance_ratio:
        result = "SINGLE_RELAY_CANDIDATE"
        relay = str(candidates[0][1])
    else:
        result = "MULTI_RELAY_OR_ECMP"
        relay = ",".join(str(dev) for _, dev in candidates)
    return result, relay, threshold, traffic, candidates


def analyze_scan(args):
    scan_dir = Path(args.scan_dir)
    with (scan_dir / "pairs.tsv").open(newline="") as handle:
        pairs = list(csv.DictReader(handle, delimiter="\t"))
    successful = [
        row for row in pairs
        if row.get("urma_status", row["status"]) == "PASS"
        and row.get("hccn_status", row["status"]) == "PASS"
    ]
    if not successful:
        raise RuntimeError(
            f"no successful EID-pair run in {scan_dir}; inspect pairs.tsv and client/server logs"
        )
    all_rows = []
    for pair in successful:
        all_rows.extend(load_delta(Path(pair["pair_dir"]) / "hccn_counter_deltas.tsv"))
    names = {row["counter"] for row in all_rows}
    tx_counter = choose_counter(names, args.tx_counter, PREFERRED_TX, "tx")
    rx_counter = choose_counter(names, args.rx_counter, PREFERRED_RX, "rx")

    grouped = defaultdict(list)
    detail_rows = []
    for pair in successful:
        rows = load_delta(Path(pair["pair_dir"]) / "hccn_counter_deltas.tsv")
        result, relay, threshold, traffic, candidates = classify(
            rows, args.endpoints, tx_counter, rx_counter,
            args.min_count, args.min_ratio, args.dominance_ratio,
        )
        key = (
            pair["src_dev"], pair["src_eid_idx"],
            pair["dst_dev"], pair["dst_eid_idx"],
        )
        grouped[key].append((result, relay))
        detail_rows.append({
            **pair,
            "classification": result,
            "relay_candidates": relay,
            "threshold": threshold,
            "candidate_scores": json.dumps(candidates),
            "device_traffic": json.dumps(traffic, sort_keys=True),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        fields = list(detail_rows[0].keys()) if detail_rows else ["classification"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(detail_rows)

    stable_rows = []
    for (src_dev, src_eid, dst_dev, dst_eid), values in sorted(grouped.items()):
        single_relays = [relay for result, relay in values if result == "SINGLE_RELAY_CANDIDATE"]
        stable = len(single_relays) == len(values) and len(set(single_relays)) == 1
        stable_rows.append({
            "src_dev": src_dev,
            "src_eid_idx": src_eid,
            "dst_dev": dst_dev,
            "dst_eid_idx": dst_eid,
            "successful_repeats": len(values),
            "stable_single_relay": int(stable),
            "relay_phy": single_relays[0] if stable else "",
            "repeat_results": ";".join(f"{result}:{relay}" for result, relay in values),
        })
    stable_output = output.with_name("stable_eid_pair_relay_map.tsv")
    with stable_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stable_rows[0].keys() if stable_rows else (
            "src_dev", "src_eid_idx", "dst_dev", "dst_eid_idx",
            "successful_repeats", "stable_single_relay",
            "relay_phy", "repeat_results"), delimiter="\t")
        writer.writeheader()
        writer.writerows(stable_rows)

    summary = {
        "tx_counter": tx_counter,
        "rx_counter": rx_counter,
        "endpoint_devices": sorted(args.endpoints),
        "pair_runs": len(pairs),
        "successful_pair_runs": len(successful),
        "stable_single_relay_pairs": [row for row in stable_rows if row["stable_single_relay"]],
        "detail_tsv": str(output.resolve()),
        "stable_map_tsv": str(stable_output.resolve()),
    }
    summary_path = output.with_name("eid_pair_relay_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("EID_PAIR_RELAY_SUMMARY " + json.dumps(summary, sort_keys=True))


def main():
    args = parse_args()
    if args.snapshot_dir:
        snapshot_delta(Path(args.snapshot_dir), Path(args.output))
        return
    analyze_scan(args)


if __name__ == "__main__":
    main()
