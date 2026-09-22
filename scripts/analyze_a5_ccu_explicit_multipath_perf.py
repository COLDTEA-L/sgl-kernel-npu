#!/usr/bin/env python3
"""Summarize native-candidate and explicit multipath AllToAll runs."""

import argparse
import json
import re
import statistics
from pathlib import Path


RESULT_RE = re.compile(r"^RESULT_JSON (\{.*\})$", re.MULTILINE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    samples = {}
    missing = []
    for log in sorted((args.run_dir / "cases").glob("*.log")):
        match = RESULT_RE.search(log.read_text(errors="replace"))
        if not match:
            missing.append(log.name)
            continue
        result = json.loads(match.group(1))
        case = re.sub(r"_r\d+$", "", log.stem)
        samples.setdefault(case, []).append(float(result["host_batch_avg_us"]))

    expected = ["native0", "native2", "direct_plus_2relay",
                "direct_plus_4relay", "direct_plus_6relay"]
    rows = []
    for case in expected:
        values = samples.get(case, [])
        rows.append({
            "case": case,
            "samples": len(values),
            "median_us": statistics.median(values) if values else None,
            "min_us": min(values) if values else None,
            "max_us": max(values) if values else None,
        })

    summary = {
        "complete": not missing and all(row["samples"] for row in rows),
        "missing_logs": missing,
        "cases": rows,
    }
    (args.run_dir / "explicit_multipath_perf_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# A5 two-rank explicit multipath AllToAll performance", "",
        f"- complete: `{summary['complete']}`",
        "- `native0/native2` mean one HCCL-discovered CommLink candidate; "
        "they are not UDMA route_addr_idx values.",
        "- explicit cases use one direct Channel plus the named relay Channels.",
        "- direct total bytes : all relay bytes = `2:1`.", "",
        "| case | samples | median_us | min_us | max_us |", "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value):
            return "NA" if value is None else f"{value:.3f}"
        lines.append(
            f"| {row['case']} | {row['samples']} | {fmt(row['median_us'])} | "
            f"{fmt(row['min_us'])} | {fmt(row['max_us'])} |"
        )
    if missing:
        lines += ["", "## Missing results", "", *[f"- `{item}`" for item in missing]]
    (args.run_dir / "explicit_multipath_perf_report.md").write_text("\n".join(lines) + "\n")
    print(args.run_dir / "explicit_multipath_perf_report.md")


if __name__ == "__main__":
    main()
