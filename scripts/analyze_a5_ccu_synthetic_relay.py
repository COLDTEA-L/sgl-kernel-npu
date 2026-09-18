#!/usr/bin/env python3
"""Summarize synthetic CommLink/EID relay probe results."""

import argparse
import csv
import re
from pathlib import Path


TRACE_RE = re.compile(
    r"SYNTHETIC_COMMLINK_TRACE rank=(\d+) peer=(\d+).*?protocol=(\d+) hop=(\d+) "
    r"die_id=(\d+) local_query_status=(\d+).*?remote_query_status=(\d+).*?"
    r"local_addr=(\S+) remote_addr=(\S+)"
)
ACQUIRE_RE = re.compile(r"HcclChannelAcquire end: status=(\d+)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    status_file = args.run_dir / "case_status.tsv"
    rows = []
    with status_file.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            log_path = args.run_dir / (row["case"] + ".log")
            text = log_path.read_text(errors="replace") if log_path.exists() else ""
            traces = TRACE_RE.findall(text)
            acquire = ACQUIRE_RE.findall(text)
            rows.append({
                **row,
                "trace_ranks": ",".join(item[0] for item in traces),
                "protocols": ",".join(item[2] for item in traces),
                "hops": ",".join(item[3] for item in traces),
                "local_query_status": ",".join(item[5] for item in traces),
                "remote_query_status": ",".join(item[6] for item in traces),
                "acquire_status": ",".join(acquire),
                "data_pass": "YES" if "PASS engine=CCU" in text else "NO",
            })

    fields = list(rows[0]) if rows else []
    with (args.run_dir / "synthetic_relay_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    passed = [row for row in rows if row["data_pass"] == "YES"]
    report = [
        "# A5 synthetic CommLink relay probe", "",
        f"- cases: `{len(rows)}`", f"- passed: `{len(passed)}`", "",
        "## Interpretation", "",
    ]
    if passed:
        report += [
            "At least one EID pair constructed from the topology inventory was accepted by",
            "`HcclChannelAcquire` and completed the data check. This proves that a candidate does not",
            "have to be returned by `HcclRankGraphGetLinks` before it can be used. HCCN counters are still",
            "required before naming the physical relay.",
        ]
    else:
        report += [
            "No synthetic EID pair completed. A catalog entry alone is therefore insufficient in this",
            "environment, or the selected EID pair/plane is not a valid provisioned pair. Inspect",
            "`local_query_status`, `remote_query_status`, and `acquire_status`; do not modify global UBUS",
            "routes based on this result.",
        ]
    (args.run_dir / "synthetic_relay_report.md").write_text("\n".join(report) + "\n")
    print(args.run_dir / "synthetic_relay_report.md")


if __name__ == "__main__":
    main()
