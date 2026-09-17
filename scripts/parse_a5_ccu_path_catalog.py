#!/usr/bin/env python3
"""Convert PATH_CATALOG records into a rank-consistent discovered-path manifest."""

import argparse
import json
import re
from pathlib import Path


LINE = re.compile(r"^PATH_CATALOG\s+(.*)$")


def parse_fields(text):
    result = {}
    for token in text.split():
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    seen = set()
    for line in args.log.read_text(errors="replace").splitlines():
        match = LINE.match(line.strip())
        if not match:
            continue
        row = parse_fields(match.group(1))
        key = (row.get("rank"), row.get("path_uid"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no PATH_CATALOG records in {args.log}")

    ranks = sorted({row["rank"] for row in rows}, key=int)
    by_rank = {rank: {row["path_uid"]: row for row in rows if row["rank"] == rank}
               for rank in ranks}
    common = set.intersection(*(set(items) for items in by_rank.values()))
    if len(ranks) != 2 or not common:
        raise RuntimeError(f"catalog is not rank-consistent: ranks={ranks}, common={sorted(common)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["path_uid", "rank0_ordinal", "rank1_ordinal", "layer", "hop",
               "protocol", "rank0_src_die", "rank1_src_die", "rank0_src_addr",
               "rank0_dst_addr"]
    with args.output.open("w") as handle:
        handle.write("\t".join(columns) + "\n")
        for uid in sorted(common):
            left, right = by_rank[ranks[0]][uid], by_rank[ranks[1]][uid]
            values = [uid, left["ordinal"], right["ordinal"], left["layer"],
                      left["hop"], left["protocol"], left["src_die"],
                      right["src_die"], left["src_addr"], left["dst_addr"]]
            handle.write("\t".join(values) + "\n")
    print(json.dumps({"catalog_tsv": str(args.output.resolve()),
                      "path_uids": sorted(common), "ranks": ranks}, sort_keys=True))


if __name__ == "__main__":
    main()
