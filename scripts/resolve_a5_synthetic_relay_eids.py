#!/usr/bin/env python3
"""Resolve src -> relay -> dst from current-machine explicit port bindings."""
import argparse

from a5_ccu_machine_topology import add_machine_arguments, load_machine, relay_candidates

FIELDS = (
    "plane", "src_die", "dst_die", "relay_die", "src_port", "relay_port_from_src",
    "relay_port_to_dst", "dst_port", "src_udmac", "dst_udmac",
    "src_eid_idx", "dst_eid_idx", "src_eid", "dst_eid",
    "src_edge", "dst_edge", "protocols",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_machine_arguments(parser)
    parser.add_argument("--src-phy", required=True, type=int)
    parser.add_argument("--dst-phy", required=True, type=int)
    parser.add_argument("--relay-phy", required=True, type=int)
    parser.add_argument("--plane", choices=("0", "1", "all"), default="all",
                        help="relay IO die, not rootinfo plane_id")
    args = parser.parse_args()
    try:
        devices = {args.src_phy, args.dst_phy, args.relay_phy}
        if len(devices) != 3:
            raise ValueError("src, dst and relay physical devices must be distinct")
        entries, edges, _ = load_machine(args, devices)
        rows = relay_candidates(entries, edges, args.src_phy, args.dst_phy,
                                args.relay_phy, args.plane, args.net_layer)
        if not rows:
            raise ValueError("no verified same-relay-die path; check explicit rootinfo EID/ports, "
                             "network plane, Atlas UB_CTP edges and live hccn EID/port status")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print("\t".join(FIELDS))
    for row in rows:
        print("\t".join(str(row[field]) for field in FIELDS))


if __name__ == "__main__":
    main()
