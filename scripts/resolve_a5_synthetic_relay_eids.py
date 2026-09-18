#!/usr/bin/env python3
"""Resolve matching source/destination EIDs for an explicit A5 relay candidate."""

import argparse
import re
from pathlib import Path


DEVICE_RE = re.compile(r"^===== Physical Device (\d+) =====$")
ROW_RE = re.compile(
    r"^\|\s*(?P<name>udmac\S*)?\s*\|\s*(?P<index>\d+)\s*\|\s*"
    r"(?P<eid>(?:[0-9a-fA-F]{4}:){7}[0-9a-fA-F]{4})\s*\|$"
)
NAME_RE = re.compile(r"^udmac(?P<instance>\d+)d(?P<die>\d+)e(?P<engine>\d+)$")


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
            "route_key": int(eid.split(":")[2], 16),
        })
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--src-phy", required=True, type=int)
    parser.add_argument("--dst-phy", required=True, type=int)
    parser.add_argument("--relay-phy", required=True, type=int)
    parser.add_argument("--plane", choices=("0", "1", "all"), default="all")
    args = parser.parse_args()

    if len({args.src_phy, args.dst_phy, args.relay_phy}) != 3:
        raise SystemExit("src, dst and relay physical devices must be distinct")
    entries = load_entries(args.topology)
    planes = (0, 1) if args.plane == "all" else (int(args.plane),)
    results = []
    for plane in planes:
        route_key = args.relay_phy + (0x40 if plane == 1 else 0)
        src = [item for item in entries
               if item["device"] == args.src_phy and item["route_key"] == route_key]
        dst = [item for item in entries
               if item["device"] == args.dst_phy and item["route_key"] == route_key]
        for left in src:
            for right in dst:
                if (left["name"], left["die"], left["engine"]) != (
                        right["name"], right["die"], right["engine"]):
                    continue
                results.append((plane, route_key, left, right))

    print("plane\troute_key\tdie\tudmac\tsrc_eid_idx\tdst_eid_idx\tsrc_eid\tdst_eid")
    for plane, route_key, src, dst in results:
        print("\t".join((str(plane), f"0x{route_key:02x}", str(src["die"]), src["name"],
                         str(src["index"]), str(dst["index"]), src["eid"], dst["eid"])))
    if not results:
        raise SystemExit(
            f"no matching EID pair for src={args.src_phy} dst={args.dst_phy} "
            f"relay={args.relay_phy} plane={args.plane}"
        )


if __name__ == "__main__":
    main()
