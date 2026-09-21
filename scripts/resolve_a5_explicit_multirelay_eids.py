#!/usr/bin/env python3
"""Resolve one unambiguous EID pair for every explicitly named relay card."""

import argparse
from pathlib import Path

from resolve_a5_synthetic_relay_eids import (
    directed_links,
    endpoint_eids,
    load_entries,
    load_topology,
)


FIELDS = (
    "route_id", "src_phy", "dst_phy", "relay_phy", "weight", "plane", "src_die", "dst_die",
    "relay_die", "src_port", "relay_port_from_src", "relay_port_to_dst",
    "dst_port", "src_udmac", "dst_udmac", "src_eid_idx", "dst_eid_idx",
    "src_eid", "dst_eid", "src_edge", "dst_edge", "protocols",
)


def csv_ints(text, name):
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be a comma-separated integer list") from exc
    if not values or len(values) != len(set(values)):
        raise SystemExit(f"{name} must be non-empty and contain no duplicates")
    return values


def resolve_relay(entries, edges, src_phy, dst_phy, relay_phy, plane, net_layer):
    src_links = directed_links(edges, src_phy, relay_phy, net_layer)
    dst_links = directed_links(edges, dst_phy, relay_phy, net_layer)
    if not src_links or not dst_links:
        raise SystemExit(
            f"relay {relay_phy}: missing UB_CTP PEER2PEER edge for "
            f"{src_phy}<->{relay_phy} or {relay_phy}<->{dst_phy}"
        )
    relay_dies = (0, 1) if plane == "all" else (int(plane),)
    results = []
    seen = set()
    for src_link in src_links:
        for dst_link in dst_links:
            src_die, src_port = src_link["endpoint_port"]
            dst_die, dst_port = dst_link["endpoint_port"]
            relay_src_die, relay_src_port = src_link["relay_port"]
            relay_dst_die, relay_dst_port = dst_link["relay_port"]
            if relay_src_die != relay_dst_die or relay_src_die not in relay_dies:
                continue
            for src_eid in endpoint_eids(entries, src_phy, src_die, src_port):
                for dst_eid in endpoint_eids(entries, dst_phy, dst_die, dst_port):
                    row = {
                        "relay_phy": relay_phy,
                        "plane": relay_src_die,
                        "src_die": src_die,
                        "dst_die": dst_die,
                        "relay_die": relay_src_die,
                        "src_port": src_port,
                        "relay_port_from_src": relay_src_port,
                        "relay_port_to_dst": relay_dst_port,
                        "dst_port": dst_port,
                        "src_udmac": src_eid["name"],
                        "dst_udmac": dst_eid["name"],
                        "src_eid_idx": src_eid["index"],
                        "dst_eid_idx": dst_eid["index"],
                        "src_eid": src_eid["eid"],
                        "dst_eid": dst_eid["eid"],
                        "src_edge": src_link["edge"],
                        "dst_edge": dst_link["edge"],
                        "protocols": ",".join(sorted(set(src_link["protocols"]) &
                                                       set(dst_link["protocols"]))),
                    }
                    identity = tuple(row.get(field) for field in FIELDS[5:])
                    if identity not in seen:
                        seen.add(identity)
                        results.append(row)
    if not results:
        raise SystemExit(
            f"relay {relay_phy}: no same-relay-die EID pair for "
            f"{src_phy}->{relay_phy}->{dst_phy}"
        )
    if len(results) != 1:
        details = ", ".join(
            f"relay_die={row['relay_die']} src={row['src_die']}/{row['src_port']} "
            f"dst={row['dst_die']}/{row['dst_port']}"
            for row in results
        )
        raise SystemExit(
            f"relay {relay_phy}: {len(results)} candidates remain ({details}); "
            "use --relay-planes to select the relay die explicitly"
        )
    return results[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--topology-json", required=True, type=Path)
    parser.add_argument("--src-phy", required=True, type=int)
    parser.add_argument("--dst-phy", required=True, type=int)
    parser.add_argument("--relay-phys", required=True,
                        help="explicit ordered relay list, for example 4,5")
    parser.add_argument("--relay-planes", default="",
                        help="one all|0|1 value per relay; default is all for every relay")
    parser.add_argument("--weights", default="",
                        help="one positive split weight per relay; default is equal weights")
    parser.add_argument("--net-layer", type=int, default=0)
    args = parser.parse_args()

    relays = csv_ints(args.relay_phys, "--relay-phys")
    if len(relays) < 2:
        raise SystemExit("multi-relay validation requires at least two explicit relay cards")
    if args.src_phy == args.dst_phy or args.src_phy in relays or args.dst_phy in relays:
        raise SystemExit("src, dst and every relay card must be distinct")
    planes = [item.strip() for item in args.relay_planes.split(",") if item.strip()]
    if not planes:
        planes = ["all"] * len(relays)
    if len(planes) != len(relays) or any(item not in ("all", "0", "1") for item in planes):
        raise SystemExit("--relay-planes must contain one all|0|1 value per relay")
    weights = csv_ints(args.weights, "--weights") if args.weights else [1] * len(relays)
    if len(weights) != len(relays) or any(value <= 0 for value in weights):
        raise SystemExit("--weights must contain one positive integer per relay")

    entries = load_entries(args.topology)
    peers, edges = load_topology(args.topology_json)
    missing = {args.src_phy, args.dst_phy, *relays} - peers
    if missing:
        raise SystemExit(f"physical device(s) absent from topology: {sorted(missing)}")

    rows = []
    for relay, plane, weight in zip(relays, planes, weights):
        row = resolve_relay(entries, edges, args.src_phy, args.dst_phy, relay,
                            plane, args.net_layer)
        row["route_id"] = 1000 + relay
        row["src_phy"] = args.src_phy
        row["dst_phy"] = args.dst_phy
        row["weight"] = weight
        rows.append(row)
    local_dies = {row["src_die"] for row in rows}
    remote_dies = {row["dst_die"] for row in rows}
    if len(local_dies) != 1 or len(remote_dies) != 1:
        raise SystemExit(
            "selected explicit relays span multiple endpoint IO dies: "
            f"src_dies={sorted(local_dies)} dst_dies={sorted(remote_dies)}; "
            "the current CCU launch supports one local die per rank"
        )

    print("\t".join(FIELDS))
    for row in rows:
        print("\t".join(str(row[field]) for field in FIELDS))


if __name__ == "__main__":
    main()
