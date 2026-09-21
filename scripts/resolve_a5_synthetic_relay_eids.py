#!/usr/bin/env python3
"""Resolve endpoint EIDs for src -> explicit relay -> dst on Atlas 950.

The driver topology JSON is authoritative for physical-device/port adjacency.
The hccn inventory translates a local (device, die, port) into the 128-bit EID
registered on that local UDMA endpoint.
"""

import argparse
import json
import re
from pathlib import Path


DEVICE_RE = re.compile(r"^===== Physical Device (\d+) =====$")
ROW_RE = re.compile(
    r"^\|\s*(?P<name>udmac\S*)?\s*\|\s*(?P<index>\d+)\s*\|\s*"
    r"(?P<eid>(?:[0-9a-fA-F]{4}:){7}[0-9a-fA-F]{4})\s*\|$"
)
NAME_RE = re.compile(r"^udmac(?P<instance>\d+)d(?P<die>\d+)e(?P<engine>\d+)$")
PORT_RE = re.compile(r"^(?P<die>\d+)/(?P<port>\d+)$")


def parse_port(text):
    match = PORT_RE.match(str(text))
    if not match:
        raise ValueError(f"invalid topology port {text!r}; expected DieID/PortID")
    return int(match.group("die")), int(match.group("port"))


def load_entries(path):
    entries = []
    device = None
    current_name = None
    for line in path.read_text(errors="replace").splitlines():
        match = DEVICE_RE.match(line.strip())
        if match:
            device = int(match.group(1))
            current_name = None
            continue
        match = ROW_RE.match(line)
        if not match or device is None:
            continue
        if match.group("name"):
            current_name = match.group("name")
        if current_name is None:
            continue
        name_match = NAME_RE.match(current_name)
        if not name_match:
            continue
        eid = match.group("eid").lower()
        entries.append({
            "device": device,
            "name": current_name,
            "instance": int(name_match.group("instance")),
            "die": int(name_match.group("die")),
            "engine": int(name_match.group("engine")),
            "index": int(match.group("index")),
            "eid": eid,
            # Mesh EIDs observed in this A5 driver inventory encode local port
            # p as p on die0 and 0x40+p on die1.  The physical peer must come
            # from edge_list; this value is not a physical-device ID.
            "port_key": int(eid.split(":")[2], 16),
        })
    return entries


def load_topology(path):
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot parse topology JSON {path}: {exc}") from exc
    peers = {int(item["local_id"]) for item in data.get("peer_list", [])}
    edges = data.get("edge_list", [])
    if int(data.get("peer_count", -1)) != len(peers):
        raise SystemExit("topology peer_count does not match peer_list")
    if int(data.get("edge_count", -1)) != len(edges):
        raise SystemExit(
            f"topology JSON is incomplete: edge_count={data.get('edge_count')} "
            f"but parsed {len(edges)} edge(s)"
        )
    return peers, edges


def pair_ports(left, right, edge_name):
    left_ports = [parse_port(value) for value in left]
    right_ports = [parse_port(value) for value in right]
    if len(left_ports) != len(right_ports):
        raise SystemExit(
            f"{edge_name} has unpaired endpoint ports: {left!r} versus {right!r}"
        )
    return list(zip(left_ports, right_ports))


def directed_links(edges, endpoint, relay, net_layer):
    links = []
    for ordinal, edge in enumerate(edges):
        if int(edge.get("net_layer", -1)) != net_layer:
            continue
        if edge.get("link_type") != "PEER2PEER":
            continue
        protocols = tuple(edge.get("protocols", []))
        if "UB_CTP" not in protocols:
            continue
        local_a = int(edge.get("local_a", -1))
        local_b = int(edge.get("local_b", -1))
        if local_a == endpoint and local_b == relay:
            pairs = pair_ports(edge.get("local_a_ports", []),
                               edge.get("local_b_ports", []), f"edge[{ordinal}]")
        elif local_b == endpoint and local_a == relay:
            pairs = pair_ports(edge.get("local_b_ports", []),
                               edge.get("local_a_ports", []), f"edge[{ordinal}]")
        else:
            continue
        for endpoint_port, relay_port in pairs:
            links.append({
                "edge": ordinal,
                "endpoint_port": endpoint_port,
                "relay_port": relay_port,
                "protocols": protocols,
            })
    return links


def endpoint_eids(entries, device, die, port):
    port_key = port + (0x40 if die == 1 else 0)
    return [item for item in entries
            if item["device"] == device and item["die"] == die
            and item["port_key"] == port_key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True, type=Path,
                        help="hccn physical-device/UDMAC/EID inventory")
    parser.add_argument("--topology-json", required=True, type=Path,
                        help="driver topology JSON containing peer edges and Die/Port pairs")
    parser.add_argument("--src-phy", required=True, type=int)
    parser.add_argument("--dst-phy", required=True, type=int)
    parser.add_argument("--relay-phy", required=True, type=int)
    parser.add_argument("--plane", choices=("0", "1", "all"), default="all",
                        help="filter the relay IO die used by both forwarding hops")
    parser.add_argument("--net-layer", type=int, default=0)
    args = parser.parse_args()

    if len({args.src_phy, args.dst_phy, args.relay_phy}) != 3:
        raise SystemExit("src, dst and relay physical devices must be distinct")
    entries = load_entries(args.topology)
    peers, edges = load_topology(args.topology_json)
    missing = {args.src_phy, args.dst_phy, args.relay_phy} - peers
    if missing:
        raise SystemExit(f"physical device(s) absent from topology peer_list: {sorted(missing)}")

    src_links = directed_links(edges, args.src_phy, args.relay_phy, args.net_layer)
    dst_links = directed_links(edges, args.dst_phy, args.relay_phy, args.net_layer)
    if not src_links:
        raise SystemExit(
            f"no net-layer {args.net_layer} UB_CTP PEER2PEER edge for "
            f"{args.src_phy}<->{args.relay_phy}"
        )
    if not dst_links:
        raise SystemExit(
            f"no net-layer {args.net_layer} UB_CTP PEER2PEER edge for "
            f"{args.dst_phy}<->{args.relay_phy}"
        )

    requested_relay_dies = (0, 1) if args.plane == "all" else (int(args.plane),)
    results = []
    for src_link in src_links:
        for dst_link in dst_links:
            src_die, src_port = src_link["endpoint_port"]
            dst_die, dst_port = dst_link["endpoint_port"]
            relay_src_die, relay_src_port = src_link["relay_port"]
            relay_dst_die, relay_dst_port = dst_link["relay_port"]
            # IO-die-only forwarding requires ingress and egress on one relay die.
            if relay_src_die != relay_dst_die or relay_src_die not in requested_relay_dies:
                continue
            src_eids = endpoint_eids(entries, args.src_phy, src_die, src_port)
            dst_eids = endpoint_eids(entries, args.dst_phy, dst_die, dst_port)
            for src_eid in src_eids:
                for dst_eid in dst_eids:
                    results.append({
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
                    })

    fields = (
        "plane", "src_die", "dst_die", "relay_die", "src_port", "relay_port_from_src",
        "relay_port_to_dst", "dst_port", "src_udmac", "dst_udmac",
        "src_eid_idx", "dst_eid_idx", "src_eid", "dst_eid",
        "src_edge", "dst_edge", "protocols",
    )
    print("\t".join(fields))
    seen = set()
    for result in results:
        identity = tuple(result[field] for field in fields[:-3])
        if identity in seen:
            continue
        seen.add(identity)
        print("\t".join(str(result[field]) for field in fields))
    if not seen:
        raise SystemExit(
            f"no same-relay-die EID pair for "
            f"{args.src_phy}->{args.relay_phy}->{args.dst_phy}; inspect both physical edges "
            "and the local EID inventory"
        )


if __name__ == "__main__":
    main()
