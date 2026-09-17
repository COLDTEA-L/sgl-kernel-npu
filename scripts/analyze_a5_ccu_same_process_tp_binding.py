#!/usr/bin/env python3
"""Summarize path-labelled URMA TP activation in sequential same-process acquires."""

import argparse
import csv
import json
import re
from pathlib import Path


LABEL_PART = re.compile(r"([^=;]+)=([^;]*)")


def parse_label(value):
    return dict(LABEL_PART.findall(value or ""))


def hex_handle(value):
    return f"0x{value:x}" if isinstance(value, int) else "N/A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for case_dir in sorted(args.run_dir.glob("order_*_r*")):
        match = re.fullmatch(r"order_(.+)_r([0-9]+)", case_dir.name)
        if match is None:
            continue
        order = match.group(1).replace("_", ",")
        repeat = int(match.group(2))
        grouped = {}
        for trace in case_dir.glob("urma_tp.pid*.jsonl"):
            for line in trace.read_text(errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") != "TP_ACTIVATION" or not event.get("trace_label"):
                    continue
                label = event["trace_label"]
                key = (label, event.get("remote_eid_raw", event.get("target_eid_raw", "N/A")))
                item = grouped.setdefault(key, {"handles": set(), "tpns": set(),
                                                "target_ids": set(), "ops": set(), "count": 0})
                if isinstance(event.get("active_tp_handle"), int):
                    item["handles"].add(event["active_tp_handle"])
                if isinstance(event.get("target_tpn"), int):
                    item["tpns"].add(event["target_tpn"])
                if isinstance(event.get("target_id"), int):
                    item["target_ids"].add(event["target_id"])
                item["ops"].add(event.get("op", "N/A"))
                item["count"] += 1
        for (label, eid), item in sorted(grouped.items()):
            parts = parse_label(label)
            rows.append({
                "case": case_dir.name,
                "order": order,
                "repeat": repeat,
                "acquire_order": parts.get("acquire_order", "N/A"),
                "ordinal": parts.get("ordinal", "N/A"),
                "path_uid": parts.get("path_uid", "N/A"),
                "remote_eid": eid,
                "active_tp_handles": ",".join(hex_handle(v) for v in sorted(item["handles"])),
                "target_tpns": ",".join(hex_handle(v) for v in sorted(item["tpns"])),
                "target_ids": ",".join(str(v) for v in sorted(item["target_ids"])),
                "ops": ",".join(sorted(item["ops"])),
                "activation_events": item["count"],
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case", "order", "repeat", "acquire_order", "ordinal", "path_uid",
        "remote_eid", "active_tp_handles", "target_tpns", "target_ids", "ops",
        "activation_events",
    ]
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    if not rows:
        (args.output.parent / "tp_binding_warning.txt").write_text(
            "No TP_ACTIVATION event carried a per-acquire trace_label. "
            "Inspect raw JSONL and the unified forensic report; do not infer path-to-TP binding.\n"
        )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
