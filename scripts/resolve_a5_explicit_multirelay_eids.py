#!/usr/bin/env python3
"""Resolve one unambiguous EID pair for every explicitly named relay card."""

import argparse

from a5_ccu_machine_topology import add_machine_arguments, load_machine, relay_candidates


FIELDS = (
    "route_id", "src_phy", "dst_phy", "relay_phy", "weight", "plane", "src_die", "dst_die",
    "relay_die", "src_port", "relay_port_from_src", "relay_port_to_dst",
    "dst_port", "src_udmac", "dst_udmac", "src_eid_idx", "dst_eid_idx",
    "src_eid", "dst_eid", "src_edge", "dst_edge", "protocols",
)


def csv_ints(text, name, *, require_unique=False):
    items = [item.strip() for item in text.split(",")]
    if not items or any(not item for item in items):
        raise SystemExit(f"{name} must be a non-empty comma-separated integer list")
    try:
        values = [int(item) for item in items]
    except ValueError as exc:
        raise SystemExit(f"{name} must be a comma-separated integer list") from exc
    if require_unique and len(values) != len(set(values)):
        raise SystemExit(f"{name} must contain no duplicates")
    return values


def resolve_relay(entries, edges, src_phy, dst_phy, relay_phy, plane, net_layer):
    results = relay_candidates(entries, edges, src_phy, dst_phy, relay_phy, plane, net_layer)
    if not results:
        raise SystemExit(
            f"relay {relay_phy}: no same-relay-die EID pair for "
            f"{src_phy}->{relay_phy}->{dst_phy}; check rootinfo EID/ports, plane, "
            "Atlas UB_CTP edges and live hccn EID/port status"
        )
    if len(results) != 1:
        details = ", ".join(
            f"relay_die={row['relay_die']} src={row['src_die']}/{row['src_port']} "
            f"dst={row['dst_die']}/{row['dst_port']}"
            for row in results
        )
        raise SystemExit(
            f"relay {relay_phy}: {len(results)} candidates remain ({details}); "
            "use --relay-planes to select the relay IO die and --plane-id for the network plane"
        )
    return results[0]


def main():
    parser = argparse.ArgumentParser()
    add_machine_arguments(parser)
    parser.add_argument("--src-phy", required=True, type=int)
    parser.add_argument("--dst-phy", required=True, type=int)
    parser.add_argument("--relay-phys", required=True,
                        help="explicit ordered relay list, for example 4,5")
    parser.add_argument("--relay-planes", default="",
                        help="one all|0|1 value per relay; default is all for every relay")
    parser.add_argument("--weights", default="",
                        help="one positive split weight per relay; default is equal weights")
    args = parser.parse_args()

    relays = csv_ints(args.relay_phys, "--relay-phys", require_unique=True)
    if len(relays) < 1:
        raise SystemExit("explicit relay validation requires at least one relay card")
    if len(relays) > 63:
        raise SystemExit("one prepared CCU plan supports at most 63 relay cards plus one direct path")
    if args.src_phy == args.dst_phy or args.src_phy in relays or args.dst_phy in relays:
        raise SystemExit("src, dst and every relay card must be distinct")
    planes = [item.strip() for item in args.relay_planes.split(",") if item.strip()]
    if not planes:
        planes = ["all"] * len(relays)
    if len(planes) != len(relays) or any(item not in ("all", "0", "1") for item in planes):
        raise SystemExit("--relay-planes must contain one all|0|1 value per relay")
    # Equal weights such as 1,1 are intentional: unlike physical relay IDs,
    # split weights do not have to be unique.
    weights = csv_ints(args.weights, "--weights") if args.weights else [1] * len(relays)
    if len(weights) != len(relays) or any(value <= 0 for value in weights):
        raise SystemExit("--weights must contain one positive integer per relay")

    try:
        entries, edges, _ = load_machine(args, {args.src_phy, args.dst_phy, *relays})
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))

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
