#!/usr/bin/env python3
"""Create one arbitrary-size relay plan plus optional 2/4/6 benchmark prefixes."""

import argparse
import csv
import json
from pathlib import Path


COUNTS = (2, 4, 6)


def read_tsv(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        return reader.fieldnames or [], rows


def write_tsv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--direct-route", type=int, default=0)
    parser.add_argument("--direct-relay-ratio", default="2:1")
    args = parser.parse_args()

    fields, rows = read_tsv(args.resolved_manifest)
    if not 1 <= len(rows) <= 63:
        raise SystemExit(f"expected 1..63 explicit relay rows, got {len(rows)}")
    relays = [int(row["relay_phy"]) for row in rows]
    if len(set(relays)) != len(rows):
        raise SystemExit("the relay rows must name distinct physical devices")
    endpoint_pairs = {(row["src_die"], row["dst_die"]) for row in rows}
    if len(endpoint_pairs) != 1:
        raise SystemExit(f"relay paths span endpoint IO dies: {sorted(endpoint_pairs)}")
    try:
        direct_ratio, relay_ratio = [int(item) for item in args.direct_relay_ratio.split(":")]
    except (ValueError, TypeError):
        raise SystemExit("--direct-relay-ratio must use positive N:M syntax") from None
    if direct_ratio <= 0 or relay_ratio <= 0:
        raise SystemExit("--direct-relay-ratio values must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plans = {}
    all_path = args.output_dir / "direct_plus_relays.tsv"
    write_tsv(all_path, fields, rows)
    plans["all"] = {
        "manifest": str(all_path.resolve()),
        "relays": relays,
        "direct_route": args.direct_route,
        "weights": [direct_ratio * len(rows), *([relay_ratio] * len(rows))],
    }
    for count in (value for value in COUNTS if value <= len(rows)):
        selected = rows[:count]
        path = args.output_dir / f"direct_plus_{count}relay.tsv"
        write_tsv(path, fields, selected)
        # Each relay has relay_ratio units. The direct path gets enough units
        # for direct_total:relay_total == direct_ratio:relay_ratio.
        weights = [direct_ratio * count, *([relay_ratio] * count)]
        plans[str(count)] = {
            "manifest": str(path.resolve()),
            "relays": relays[:count],
            "direct_route": args.direct_route,
            "weights": weights,
        }
    metadata = {
        "version": 1,
        "world_size": 2,
        "direct_relay_aggregate_ratio": [direct_ratio, relay_ratio],
        "plans": plans,
    }
    (args.output_dir / "multipath_plans.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(args.output_dir / "multipath_plans.json")


if __name__ == "__main__":
    main()
