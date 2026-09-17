#!/usr/bin/env python3
"""Resolve A5 UBUS neighbors and control the experimental route override."""

import argparse
import csv
import pathlib
import re
import sys


DIRECT_LINK_RE = re.compile(r"^\s*(0x[0-9a-fA-F]+|\d+)\s*:\s*"
                            r"(0x[0-9a-fA-F]+|\d+)\s*"
                            r"\[(0x[0-9a-fA-F]+|\d+)\]\s*$")


def read_int(path: pathlib.Path) -> int:
    return int(path.read_text().strip(), 0)


def discover_entities(sysfs_root: pathlib.Path):
    entities = {}
    for route_file in sysfs_root.glob("*/route_override"):
        root = route_file.parent
        try:
            uent = read_int(root / "uent_num")
            cna = read_int(root / "primary_cna")
        except (FileNotFoundError, ValueError):
            continue
        links = []
        direct_link = root / "direct_link"
        if direct_link.exists():
            for line in direct_link.read_text().splitlines():
                match = DIRECT_LINK_RE.match(line)
                if match:
                    links.append({"port": int(match.group(1), 0),
                                  "remote_port": int(match.group(2), 0),
                                  "remote_uent": int(match.group(3), 0)})
        entities[uent] = {
            "uent": uent,
            "cna": cna,
            "path": root,
            "device": root.name,
            "entity_idx": read_int(root / "entity_idx") if (root / "entity_idx").exists() else None,
            "eid": read_int(root / "eid") if (root / "eid").exists() else None,
            "user_eid": read_int(root / "user_eid") if (root / "user_eid").exists() else None,
            "upi": read_int(root / "upi") if (root / "upi").exists() else None,
            "links": links,
        }
    if not entities:
        raise RuntimeError(f"no patched UBUS entities below {sysfs_root}")
    return entities


def load_phy_map(path: pathlib.Path):
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(line for line in handle if not line.lstrip().startswith("#")):
            phy = int(row["phy_id"], 0)
            uent = int(row["uent_num"], 0)
            if phy in result:
                raise RuntimeError(f"duplicate physical device {phy} in {path}")
            result[phy] = uent
    return result


def unique_link(entities, src_uent, dst_uent):
    links = [item for item in entities[src_uent]["links"] if item["remote_uent"] == dst_uent]
    if len(links) != 1:
        raise RuntimeError(f"expected exactly one active UBUS link {src_uent}->{dst_uent}; got {links}")
    return links[0]


def resolve(args, entities):
    mapping = load_phy_map(pathlib.Path(args.phy_map))
    requested = {"src": args.src_phy, "relay": args.relay_phy, "dst": args.dst_phy}
    resolved = {}
    for name, phy in requested.items():
        if phy not in mapping:
            raise RuntimeError(f"physical device {phy} is absent from {args.phy_map}")
        uent = mapping[phy]
        if uent not in entities:
            raise RuntimeError(f"physical device {phy} maps to missing UBUS entity {uent}")
        resolved[name] = entities[uent]
    if len({item["uent"] for item in resolved.values()}) != 3:
        raise RuntimeError("source, relay and destination must map to three distinct entities")
    first = unique_link(entities, resolved["src"]["uent"], resolved["relay"]["uent"])
    second = unique_link(entities, resolved["relay"]["uent"], resolved["dst"]["uent"])
    direct = unique_link(entities, resolved["src"]["uent"], resolved["dst"]["uent"])
    print(f"src:   phy={args.src_phy} uent={resolved['src']['uent']} cna={resolved['src']['cna']} device={resolved['src']['device']}")
    print(f"relay: phy={args.relay_phy} uent={resolved['relay']['uent']} cna={resolved['relay']['cna']} device={resolved['relay']['device']}")
    print(f"dst:   phy={args.dst_phy} uent={resolved['dst']['uent']} cna={resolved['dst']['cna']} device={resolved['dst']['device']}")
    print(f"resolved direct port: src port[{direct['port']}] -> dst uent={resolved['dst']['uent']}")
    print(f"resolved relay first hop: src port[{first['port']}] -> relay uent={resolved['relay']['uent']}")
    print(f"verified relay native second hop: relay port[{second['port']}] -> dst uent={resolved['dst']['uent']}")
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("topology", "resolve", "force", "add", "clear", "status"))
    parser.add_argument("--sysfs-root", default="/sys/bus/ub/devices")
    parser.add_argument("--phy-map")
    parser.add_argument("--src-phy", type=int)
    parser.add_argument("--relay-phy", type=int)
    parser.add_argument("--dst-phy", type=int)
    args = parser.parse_args()
    entities = discover_entities(pathlib.Path(args.sysfs_root))
    if args.command == "topology":
        for uent in sorted(entities):
            entity = entities[uent]
            print(f"uent={uent} cna={entity['cna']} device={entity['device']} "
                  f"entity_idx={entity['entity_idx']} eid={entity['eid']} "
                  f"user_eid={entity['user_eid']} upi={entity['upi']}")
            for link in entity["links"]:
                print(f"  port[{link['port']}] -> remote_uent={link['remote_uent']} "
                      f"remote_port={link['remote_port']}")
        return
    required = (args.phy_map, args.src_phy, args.relay_phy, args.dst_phy)
    if any(value is None for value in required):
        parser.error("resolve/force/add/clear/status require --phy-map, --src-phy, --relay-phy and --dst-phy")
    resolved = resolve(args, entities)
    route_file = resolved["src"]["path"] / "route_override"
    if args.command == "resolve":
        return
    if args.command == "status":
        print(route_file.read_text(), end="")
        return
    if args.command == "clear":
        payload = f"clear {resolved['dst']['uent']}\n"
    else:
        payload = f"{args.command} {resolved['relay']['uent']} {resolved['dst']['uent']}\n"
    route_file.write_text(payload)
    print(f"wrote {payload.strip()!r} to {route_file}")
    print(route_file.read_text(), end="")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
