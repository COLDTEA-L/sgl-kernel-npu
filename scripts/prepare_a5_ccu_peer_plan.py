#!/usr/bin/env python3
"""Compile an explicitly supplied physical peer policy; never allocate relays."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python/deep_ep/deep_ep"))
from ccu_l1_config import select_l1_config

from resolve_a5_explicit_multirelay_eids import resolve_relay
from a5_ccu_machine_topology import add_machine_arguments, load_machine

FIELDS = ("src_rank", "dst_rank", "src_phy", "dst_phy", "kind", "relay_phy",
          "direct_route", "weight", "src_die", "dst_die", "src_eid", "dst_eid")


def parse_phys_ids(value, option):
    parts = value.split(",")
    if any(not part.strip().isascii() or not part.strip().isdecimal() for part in parts):
        raise ValueError(f"{option} requires comma-separated nonnegative physical IDs")
    ids = [int(part.strip()) for part in parts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{option} must contain no duplicates")
    return ids


def load_requested_policy(devices, relay_map=None, relay_phys=None):
    if relay_map is not None and relay_phys is not None:
        raise ValueError("--relay-phys and --relay-map are mutually exclusive")
    if relay_phys is not None:
        if len(devices) != 2:
            raise ValueError("--relay-phys is for two-card tests; use --relay-map for per-pair four-card policies")
        relays = parse_phys_ids(relay_phys, "--relay-phys")
        return {",".join(str(device) for device in sorted(devices)): relays}
    return json.loads(relay_map.read_text()) if relay_map is not None else {}


def validate_policy(devices, available, policy):
    if len(devices) not in (2, 4, 8) or len(set(devices)) != len(devices):
        raise ValueError("--devices requires 2/4 or direct-only 8 distinct physical devices")
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
        if len(devices) == 8 and values:
            raise ValueError("8-rank plans are direct-only")
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


def build_rows(devices, available, policy, entries, edges, direct_route=0, net_layer=0):
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
            resolved = resolve_relay(entries, edges, src, dst, relay, spec["plane"], net_layer)
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
    p.add_argument("--l1-config", type=Path)
    p.add_argument("--devices", required=True)
    p.add_argument("--available-phys", default="0,1,2,3,4,5,6,7")
    relay_options = p.add_mutually_exclusive_group()
    relay_options.add_argument("--relay-map", type=Path, help="explicit physical-pair JSON policy (two or four cards)")
    relay_options.add_argument("--relay-phys", help="two-card explicit relay IDs, e.g. 0,1; omit both options for direct-only")
    p.add_argument("--direct-route", type=int, default=0)
    add_machine_arguments(p)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--validate-only", action="store_true", help="validate policy without runtime, topology reads or file writes")
    args = p.parse_args()
    if not args.validate_only and args.output_dir is None:
        p.error("--output-dir is required unless --validate-only is used")
    try:
        args.l1_config = select_l1_config(args.l1_config)
        devices = parse_phys_ids(args.devices, "--devices")
        available = parse_phys_ids(args.available_phys, "--available-phys")
        if args.direct_route < 0:
            raise ValueError("direct route must be nonnegative")
        policy = load_requested_policy(devices, args.relay_map, args.relay_phys)
        normalized, limit = validate_policy(devices, available, policy)
        if args.l1_config:
            from a5_ccu_l1_policy import validate_config
            validate_config(args.l1_config, devices, args.root_info)
        if args.validate_only:
            print("PEER_POLICY_VALIDATED " + json.dumps(dict(devices=devices,
                available_devices=available, relay_limit_per_peer=limit, relay_policy=policy)))
            return
        if args.topology is not None:
            raise ValueError("--topology TXT input has been removed; use --root-info and --hccn-tool")
        if any(normalized.values()) or args.l1_config:
            required = set(devices) | {spec["physical_device"]
                                      for specs in normalized.values() for spec in specs}
            entries, edges, provenance = load_machine(args, required)
        else:
            # Direct-only uses HCCL's current communicator path discovery. No
            # synthetic EID, old TXT, or address-bit mapping is involved.
            entries, edges = [], []
            provenance = {"source": "hccl_direct_discovery"}
        rows, summary = build_rows(devices, available, policy, entries, edges,
                                   args.direct_route, args.net_layer)
        if args.l1_config:
            from a5_ccu_machine_topology import directed_links, connected_pair
            from a5_ccu_l1_policy import apply_l1
            for row in rows:
                if row['kind'] != 'direct': continue
                matches={}
                for link in directed_links(edges,row['src_phy'],row['dst_phy'],args.net_layer):
                    for a,b in connected_pair(entries,row['src_phy'],link['endpoint_port'],row['dst_phy'],link['relay_port'],args.net_layer):
                        matches[(a['eid'],b['eid'])]=(a,b)
                if len(matches)!=1: raise ValueError('L1 needs a unique machine-bound direct path per peer')
                a,b=next(iter(matches.values()))
                row.update(src_eid=a['eid'],dst_eid=b['eid'],src_die=a['die'],dst_die=b['die'])
            rows, l1_policy=apply_l1(rows,devices,args.l1_config,provenance['root_info'])
            summary.update(l1_policy=l1_policy,rows=rows,channels_per_rank=[sum(k in (v['src_rank'],v['dst_rank']) for v in rows) for k in range(len(devices))])
            if any(9+3*n>48 for n in summary['channels_per_rank']): raise ValueError('L1 catalog exceeds CCU argument budget')
        summary["machine_topology"] = provenance
        summary["policy_source"] = ("l1_config" if args.l1_config else "command_line" if args.relay_phys is not None
                                    else "json" if args.relay_map is not None else "direct_only")
        summary["requested_relay_policy"] = policy
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
