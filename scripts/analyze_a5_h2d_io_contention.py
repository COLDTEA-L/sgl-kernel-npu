#!/usr/bin/env python3
"""Analyze solo/concurrent two-card H2D and D2H measurements."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

support = Path(__file__).resolve().parents[1] / "tests/python/deepep"
sys.path.insert(0, str(support))
from a5_h2d_io_test_support import interval_overlap  # noqa: E402


SOLO_CASE = {
    (0, "h2d"): "h2d_solo_rank0",
    (1, "h2d"): "h2d_solo_rank1",
    (0, "d2h"): "d2h_solo_rank0",
    (1, "d2h"): "d2h_solo_rank1",
}


def median(values):
    return statistics.median(values) if values else None


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    settings = json.loads((root / "run_settings.json").read_text())
    rank_results = []
    errors = []
    for rank in (0, 1):
        path = root / "results" / f"rank{rank}.json"
        if not path.is_file():
            errors.append(f"missing {path.relative_to(root)}")
            continue
        data = json.loads(path.read_text())
        if data.get("rank") != rank or data.get("world_size") != 2:
            errors.append(f"invalid rank metadata in {path.name}")
        if not data.get("pinned_host"):
            errors.append(f"rank{rank} did not use pinned host memory")
        expected_gates = 2 * len(settings["sizes"])
        gates = data.get("correctness", [])
        if len(gates) != expected_gates or any(item.get("status") != "PASS" for item in gates):
            errors.append(f"rank{rank} correctness gates incomplete")
        rank_results.append(data)

    records = []
    for result in rank_results:
        for item in result.get("records", []):
            record = dict(item)
            record["rank"] = result["rank"]
            record["physical_device"] = result.get("physical_device")
            records.append(record)

    expected_records = 2 * len(settings["sizes"]) * len(settings["cases"]) * settings["repeats"]
    if len(records) != expected_records:
        errors.append(f"record count {len(records)} != expected {expected_records}")
    active = [record for record in records if record.get("active")]
    if any(record.get("correctness") != "PASS" for record in active):
        errors.append("an active measurement lacks correctness PASS")

    raw_fields = (
        "rank", "physical_device", "case", "direction", "active", "bytes", "repeat",
        "iterations", "start_lateness_us", "started_ns", "ended_ns", "host_total_us",
        "host_avg_us", "device_total_us", "device_avg_us", "host_gbps", "device_gbps",
        "correctness",
    )
    with (root / "h2d_io_records.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=raw_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: (row["bytes"], row["case"], row["repeat"], row["rank"])))

    index = {(row["case"], row["bytes"], row["repeat"], row["rank"]): row for row in records}
    inflation_rows = []
    concurrent_cases = [name for name in settings["cases"] if name in (
        "h2d_both", "d2h_both", "mixed_rank0_h2d_rank1_d2h", "mixed_rank0_d2h_rank1_h2d")]
    for size in settings["sizes"]:
        for case_name in concurrent_cases:
            for repeat in range(1, settings["repeats"] + 1):
                pair = [index.get((case_name, size, repeat, rank)) for rank in (0, 1)]
                active_pair = [row for row in pair if row and row.get("active")]
                overlap = None
                start_skew_us = None
                if len(active_pair) == 2:
                    overlap = interval_overlap(
                        active_pair[0]["started_ns"], active_pair[0]["ended_ns"],
                        active_pair[1]["started_ns"], active_pair[1]["ended_ns"])
                    start_skew_us = abs(active_pair[0]["started_ns"] - active_pair[1]["started_ns"]) / 1000.0
                for row in active_pair:
                    baseline_name = SOLO_CASE[(row["rank"], row["direction"])]
                    baseline = index.get((baseline_name, size, repeat, row["rank"]))
                    if baseline is None or not baseline.get("active"):
                        errors.append(f"missing baseline {baseline_name} size={size} repeat={repeat} rank={row['rank']}")
                        continue
                    inflation_rows.append({
                        "bytes": size,
                        "case": case_name,
                        "repeat": repeat,
                        "rank": row["rank"],
                        "physical_device": row.get("physical_device"),
                        "direction": row["direction"],
                        "solo_device_avg_us": baseline["device_avg_us"],
                        "concurrent_device_avg_us": row["device_avg_us"],
                        "device_time_inflation": row["device_avg_us"] / baseline["device_avg_us"],
                        "device_time_inflation_percent":
                            (row["device_avg_us"] / baseline["device_avg_us"] - 1.0) * 100.0,
                        "solo_host_avg_us": baseline["host_avg_us"],
                        "concurrent_host_avg_us": row["host_avg_us"],
                        "host_time_inflation": row["host_avg_us"] / baseline["host_avg_us"],
                        "solo_device_gbps": baseline["device_gbps"],
                        "concurrent_device_gbps": row["device_gbps"],
                        "start_skew_us": start_skew_us,
                        "overlap_shorter_fraction": overlap["shorter_fraction"] if overlap else None,
                        "overlap_longer_fraction": overlap["longer_fraction"] if overlap else None,
                    })

    inflation_fields = (
        "bytes", "case", "repeat", "rank", "physical_device", "direction",
        "solo_device_avg_us", "concurrent_device_avg_us", "device_time_inflation",
        "device_time_inflation_percent", "solo_host_avg_us", "concurrent_host_avg_us",
        "host_time_inflation", "solo_device_gbps", "concurrent_device_gbps",
        "start_skew_us", "overlap_shorter_fraction", "overlap_longer_fraction",
    )
    with (root / "h2d_io_inflation.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=inflation_fields)
        writer.writeheader()
        writer.writerows(inflation_rows)

    groups = {}
    for row in inflation_rows:
        key = (row["bytes"], row["case"], row["rank"], row["physical_device"], row["direction"])
        groups.setdefault(key, []).append(row)
    aggregate = []
    for key, rows in sorted(groups.items()):
        size, case_name, rank, device, direction = key
        aggregate.append({
            "bytes": size,
            "case": case_name,
            "rank": rank,
            "physical_device": device,
            "direction": direction,
            "repeats": len(rows),
            "solo_device_avg_us_median": median([row["solo_device_avg_us"] for row in rows]),
            "concurrent_device_avg_us_median": median([row["concurrent_device_avg_us"] for row in rows]),
            "device_time_inflation_median": median([row["device_time_inflation"] for row in rows]),
            "device_time_inflation_percent_median": median([row["device_time_inflation_percent"] for row in rows]),
            "solo_device_gbps_median": median([row["solo_device_gbps"] for row in rows]),
            "concurrent_device_gbps_median": median([row["concurrent_device_gbps"] for row in rows]),
            "start_skew_us_median": median([row["start_skew_us"] for row in rows if row["start_skew_us"] is not None]),
            "overlap_shorter_fraction_median": median([
                row["overlap_shorter_fraction"] for row in rows if row["overlap_shorter_fraction"] is not None]),
            "overlap_longer_fraction_median": median([
                row["overlap_longer_fraction"] for row in rows if row["overlap_longer_fraction"] is not None]),
        })

    complete = not errors and bool(aggregate)
    summary = {
        "complete": complete,
        "errors": errors,
        "settings": settings,
        "rank_devices": {str(item["rank"]): item.get("physical_device") for item in rank_results},
        "raw_record_count": len(records),
        "inflation_record_count": len(inflation_rows),
        "aggregate": aggregate,
    }
    (root / "h2d_io_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = [
        "# A5 two-card H2D/D2H contention report", "",
        f"- complete: `{complete}`",
        f"- physical devices: `{settings['devices']}`",
        f"- pinned host memory: `{all(item.get('pinned_host') for item in rank_results)}`",
        f"- warmup / measured calls / repeats: `{settings['warmup']} / {settings['iterations']} / {settings['repeats']}`",
        "- device time is measured by NPU events on the copy stream; host time additionally includes submission and synchronize.",
        "- H2D means host read + device HBM write; D2H means device HBM read + host write.", "",
    ]
    if errors:
        report += ["## Errors", "", *[f"- {error}" for error in errors], ""]
    report += [
        "## Median concurrent/solo results", "",
        "| bytes/copy | case | physical card | direction | solo device us | concurrent device us | time inflation | concurrent GB/s | start skew us | overlap/shorter | overlap/longer |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        report.append(
            f"| {row['bytes']} | {row['case']} | {row['physical_device']} | {row['direction']} | "
            f"{fmt(row['solo_device_avg_us_median'])} | {fmt(row['concurrent_device_avg_us_median'])} | "
            f"{fmt(row['device_time_inflation_median'])}x ({fmt(row['device_time_inflation_percent_median'], 1)}%) | "
            f"{fmt(row['concurrent_device_gbps_median'])} | {fmt(row['start_skew_us_median'])} | "
            f"{fmt(row['overlap_shorter_fraction_median'])} | {fmt(row['overlap_longer_fraction_median'])} |"
        )
    report += [
        "", "## Interpretation", "",
        "- Compare `h2d_both` with each card's H2D solo row, and `d2h_both` with D2H solo.",
        "- Mixed cases are full-duplex controls; they do not replace the same-direction cases.",
        "- A ratio above 1 means time inflation. Report absolute time as well because small transfers are host/launch dominated.",
        "- This experiment diagnoses host-I/O and HBM-side contention only. It does not exercise CCU, HCCL, UB relay paths, or prove their congestion behavior.",
        "- Reject a run with incomplete correctness, large start skew, or little interval overlap; inspect the raw CSV instead of relying only on the median table.", "",
    ]
    (root / "h2d_io_report.md").write_text("\n".join(report))
    print(json.dumps({"complete": complete, "errors": errors, "aggregate_rows": len(aggregate)}))
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
