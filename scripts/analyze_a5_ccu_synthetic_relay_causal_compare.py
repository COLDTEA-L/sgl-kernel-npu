#!/usr/bin/env python3
"""Compare native and synthetic CCU path footprints from HCCN counters."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


COUNTERS = ("rx_busi_flit_num", "tx_busi_flit_num")
EXPECTED_CASES = ("native0", "synthetic0", "native2", "synthetic2")


def read_tsv(path: Path):
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_tsv(path: Path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def median(values):
    return float(statistics.median(values)) if values else 0.0


def normalized(vector, keys):
    values = [max(0.0, vector.get(key, 0.0)) for key in keys]
    total = sum(values)
    return [value / total for value in values] if total else [0.0] * len(values)


def cosine_log(a, b, keys):
    va = [math.log1p(max(0.0, a.get(key, 0.0))) for key in keys]
    vb = [math.log1p(max(0.0, b.get(key, 0.0))) for key in keys]
    denom = math.sqrt(sum(x * x for x in va) * sum(x * x for x in vb))
    return sum(x * y for x, y in zip(va, vb)) / denom if denom else 0.0


def l1_distance(a, b, keys):
    va = normalized(a, keys)
    vb = normalized(b, keys)
    return 0.5 * sum(abs(x - y) for x, y in zip(va, vb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--src-phy", required=True, type=int)
    parser.add_argument("--dst-phy", required=True, type=int)
    parser.add_argument("--relay-phy", required=True, type=int)
    args = parser.parse_args()

    status_path = args.run_dir / "case_status.tsv"
    status_rows = list(read_tsv(status_path))
    samples = defaultdict(list)
    long_rows = []
    valid_runs = defaultdict(int)

    for status in status_rows:
        tsv_text = status.get("hccn_tsv", "")
        tsv_path = Path(tsv_text) if tsv_text else None
        if status.get("result") != "PASS" or tsv_path is None or not tsv_path.is_file():
            continue
        valid_runs[status["case"]] += 1
        for row in read_tsv(tsv_path):
            if row["counter"] not in COUNTERS:
                continue
            delta = int(row["delta"])
            key = (row["physical_device"], row["udie"], row["port"], row["counter"])
            samples[(status["case"], key)].append(delta)
            long_rows.append({
                "case": status["case"],
                "repeat": status["repeat"],
                "physical_device": row["physical_device"],
                "udie": row["udie"],
                "port": row["port"],
                "counter": row["counter"],
                "delta": delta,
                "source": str(tsv_path),
            })

    write_tsv(
        args.run_dir / "footprint_long.tsv",
        ("case", "repeat", "physical_device", "udie", "port", "counter", "delta", "source"),
        long_rows,
    )

    vectors = {case: {} for case in EXPECTED_CASES}
    median_rows = []
    all_keys = set()
    for (case, key), values in samples.items():
        value = median(values)
        vectors.setdefault(case, {})[key] = value
        all_keys.add(key)
        median_rows.append({
            "case": case,
            "physical_device": key[0],
            "udie": key[1],
            "port": key[2],
            "counter": key[3],
            "median_delta": f"{value:.3f}",
            "samples": len(values),
            "min_delta": min(values),
            "max_delta": max(values),
        })
    median_rows.sort(key=lambda row: (row["case"], -float(row["median_delta"])))
    write_tsv(
        args.run_dir / "median_port_footprints.tsv",
        ("case", "physical_device", "udie", "port", "counter", "median_delta",
         "samples", "min_delta", "max_delta"),
        median_rows,
    )

    device_rows = []
    device_summary = {}
    for case in EXPECTED_CASES:
        for dev in range(8):
            rx = sum(value for key, value in vectors[case].items()
                     if int(key[0]) == dev and key[3] == "rx_busi_flit_num")
            tx = sum(value for key, value in vectors[case].items()
                     if int(key[0]) == dev and key[3] == "tx_busi_flit_num")
            balanced = min(rx, tx)
            device_summary[(case, dev)] = (rx, tx, balanced)
            device_rows.append({
                "case": case,
                "physical_device": dev,
                "rx_median_sum": f"{rx:.3f}",
                "tx_median_sum": f"{tx:.3f}",
                "balanced_relay_score": f"{balanced:.3f}",
            })
    write_tsv(
        args.run_dir / "device_traffic_summary.tsv",
        ("case", "physical_device", "rx_median_sum", "tx_median_sum", "balanced_relay_score"),
        device_rows,
    )

    keys = sorted(all_keys)
    comparison_pairs = (
        ("native0", "native2"),
        ("synthetic0", "native0"),
        ("synthetic0", "native2"),
        ("synthetic2", "native0"),
        ("synthetic2", "native2"),
        ("synthetic0", "synthetic2"),
    )
    similarity_rows = []
    metrics = {}
    for left, right in comparison_pairs:
        cosine = cosine_log(vectors[left], vectors[right], keys)
        distance = l1_distance(vectors[left], vectors[right], keys)
        metrics[(left, right)] = (cosine, distance)
        similarity_rows.append({
            "left": left,
            "right": right,
            "cosine_log": f"{cosine:.6f}",
            "normalized_l1_distance": f"{distance:.6f}",
        })
    write_tsv(
        args.run_dir / "footprint_similarity.tsv",
        ("left", "right", "cosine_log", "normalized_l1_distance"),
        similarity_rows,
    )

    missing = [case for case in EXPECTED_CASES if valid_runs[case] == 0]
    conclusion = "INCONCLUSIVE"
    reason = "The four successful HCCN footprints are required before causal classification."
    if not missing:
        baseline_distance = metrics[("native0", "native2")][1]
        s0_n0 = metrics[("synthetic0", "native0")][1]
        s0_n2 = metrics[("synthetic0", "native2")][1]
        s2_n0 = metrics[("synthetic2", "native0")][1]
        s2_n2 = metrics[("synthetic2", "native2")][1]
        synth_distance = metrics[("synthetic0", "synthetic2")][1]
        margin = 0.05
        if baseline_distance < margin:
            reason = "Native route0 and route2 footprints are not distinguishable in this run."
        elif s0_n0 + margin < s0_n2 and s2_n2 + margin < s2_n0:
            conclusion = "BASE_PATH_BOUND"
            reason = "Each synthetic footprint follows its base CommLink more closely than the fixed EID pair."
        elif synth_distance + margin < min(s0_n0, s0_n2, s2_n0, s2_n2):
            conclusion = "SYNTHETIC_EID_BOUND_CANDIDATE"
            reason = "The two synthetic footprints match each other more closely than either native baseline."
        else:
            reason = "The footprint relationships do not provide a stable causal separation."

    relay_rows = []
    for case in EXPECTED_CASES:
        rx, tx, balanced = device_summary[(case, args.relay_phy)]
        relay_rows.append((case, rx, tx, balanced))

    report = [
        "# A5 CCU synthetic relay causal comparison", "",
        f"- source: `{args.src_phy}`", f"- destination: `{args.dst_phy}`",
        f"- requested relay: `{args.relay_phy}`", f"- conclusion: **{conclusion}**",
        f"- reason: {reason}", "",
        "## Valid HCCN runs", "",
    ]
    for case in EXPECTED_CASES:
        report.append(f"- `{case}`: {valid_runs[case]}")
    report += ["", "## Requested relay traffic", "",
               "| case | RX median sum | TX median sum | balanced score |",
               "|---|---:|---:|---:|"]
    for case, rx, tx, balanced in relay_rows:
        report.append(f"| {case} | {rx:.0f} | {tx:.0f} | {balanced:.0f} |")
    report += [
        "", "## Interpretation boundary", "",
        "`BASE_PATH_BOUND` means changing the base CommLink changes the physical footprint while the",
        "synthetic EID pair stays fixed; the EID overwrite is therefore not the effective selector.",
        "`SYNTHETIC_EID_BOUND_CANDIDATE` is only a candidate conclusion: the requested relay must also",
        "show repeatable, balanced ingress/egress above its matching native baseline, while other relay",
        "devices do not. Shared-machine background traffic can make the result `INCONCLUSIVE`.", "",
        "See `median_port_footprints.tsv`, `device_traffic_summary.tsv`, and",
        "`footprint_similarity.tsv` for the evidence.",
    ]
    (args.run_dir / "causal_comparison_report.md").write_text("\n".join(report) + "\n")
    print(args.run_dir / "causal_comparison_report.md")


if __name__ == "__main__":
    main()
