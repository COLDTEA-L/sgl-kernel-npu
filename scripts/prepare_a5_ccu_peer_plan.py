#!/usr/bin/env python3
"""Compile an explicitly supplied physical peer policy; never allocate relays."""
import argparse
import csv
import itertools
import json
from pathlib import Path

from resolve_a5_explicit_multirelay_eids import resolve_relay
from resolve_a5_synthetic_relay_eids import load_entries, load_topology

FIELDS = ("src_rank", "dst_rank", "src_phy", "dst_phy", "kind", "relay_phy",
          "direct_route", "weight", "src_die", "dst_die", "src_eid", "dst_eid")


def validate_policy(devices, available, policy):
    if len(devices) not in (2, 4) or len(set(devices)) != len(devices):
        raise ValueError("--devices requires 2 or 4 distinct physical devices")
    if any(d < 0 for d in available) or len(set(available)) != len(available) or not set(devices) <= set(available):
        raise ValueError("available device set must be unique and contain communicator")
    if not isinstance(policy, dict):
        raise ValueError("relay-map must be an object keyed by physical pair, e.g. '0,1'")
    limit = (len(available) - len(devices)) // (len(devices) - 1)
    normalized, used = {}, {d: set() for d in devices}
    for key, values in policy.items():
        pair = tuple(int(x) for x in key.split(","))
        if len(pair) != 2 or pair[0] >= pair[1] or not set(pair) <= set(devices):
            raise ValueError(f"invalid canonical physical pair {key}")
        if not isinstance(values, list) or len(values) > limit:
            raise ValueError(f"pair {key}: relay count exceeds (N-k)//(k-1)={limit}")
        rows, seen = [], set()
        for value in values:
            spec = {"physical_device": value} if isinstance(value, int) else dict(value)
            relay = spec["physical_device"]
            if type(relay) is not int:
                raise ValueError(f"pair {key}: relay physical_device must be an integer")
            if relay in devices or relay not in available or relay in seen:
                raise ValueError(f"pair {key}: relay must be unique and outside communicator")
            if any(relay in used[d] for d in pair):
                raise ValueError(f"pair {key}: a source rank cannot reuse relay {relay} for different peers")
            seen.add(relay)
            for d in pair:
                used[d].add(relay)
            plane = str(spec.get("plane", "all"))
            if plane not in ("all", "0", "1"):
                raise ValueError(f"invalid plane for relay {relay}")
            rows.append({"physical_device": relay, "plane": plane})
        normalized[pair] = rows
    return normalized, limit


def build_rows(devices, available, policy, entries, edges, direct_route=0):
    policy, limit = validate_policy(devices, available, policy)
    rows, diagnostics = [], []
    local_dies = {d: set() for d in devices}
    for a, b in itertools.combinations(range(len(devices)), 2):
        src, dst = devices[a], devices[b]
        base = dict(src_rank=a, dst_rank=b, src_phy=src, dst_phy=dst,
                    direct_route=direct_route)
        rows.append(dict(base, kind="direct", relay_phy="-", weight=2,
                         src_die=0, dst_die=0, src_eid="-", dst_eid="-"))
        for spec in policy.get(tuple(sorted((src, dst))), []):
            relay = spec["physical_device"]
            resolved = resolve_relay(entries, edges, src, dst, relay, spec["plane"], 0)
            local_dies[src].add(resolved["src_die"])
            local_dies[dst].add(resolved["dst_die"])
            rows.append(dict(base, kind="relay", relay_phy=relay, weight=1,
                **{key: resolved[key] for key in ("src_die", "dst_die", "src_eid", "dst_eid")}))
            diagnostics.append(dict(base, **resolved))
    if any(len(values) > 1 for values in local_dies.values()):
        raise ValueError("selected relays span multiple local dies on a rank; initial scheduler supports one")
    channels = [sum(r["src_rank"] == rank or r["dst_rank"] == rank for r in rows)
                for rank in range(len(devices))]
    if any(4 + 3 * count > 48 for count in channels):
        raise ValueError("plan exceeds current CCU 48 launch-argument limit; never silently truncate")
    return rows, dict(devices=devices, available_devices=available, relay_limit_per_peer=limit,
                     channels_per_rank=channels, rows=rows, relay_edges=diagnostics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", required=True)
    p.add_argument("--available-phys", default="0,1,2,3,4,5,6,7")
    p.add_argument("--relay-map", type=Path)
    p.add_argument("--direct-route", type=int, default=0)
    p.add_argument("--topology", type=Path, default=Path("docs/topology/a5_hccn_device_topology_raw.txt"))
    p.add_argument("--topology-json", type=Path, default=Path("/usr/local/Ascend/driver/topo/950/atlas_950_1.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        devices = [int(x) for x in args.devices.split(",")]
        available = [int(x) for x in args.available_phys.split(",")]
        if args.direct_route < 0:
            raise ValueError("direct route must be nonnegative")
        policy = json.loads(args.relay_map.read_text()) if args.relay_map else {}
        normalized, _ = validate_policy(devices, available, policy)
        if any(normalized.values()):
            entries = load_entries(args.topology)
            known, edges = load_topology(args.topology_json)
            if not set(available) <= known:
                raise ValueError("available physical IDs missing from driver topology")
        else:
            entries, edges = [], []  # direct-only never requires an EID inventory
        rows, summary = build_rows(devices, available, policy, entries, edges, args.direct_route)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "peer_plan.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "peer_plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(args.output_dir / "peer_plan.tsv")


if __name__ == "__main__":
    main()
