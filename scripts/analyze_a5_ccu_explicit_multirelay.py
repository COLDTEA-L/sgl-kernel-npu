#!/usr/bin/env python3
"""Prepare and analyze strict explicit-relay CCU multi-path experiments."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


RESULT_RE = re.compile(r"^RESULT_JSON (\{.*\})$", re.MULTILINE)


def read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def prepare(run_dir, manifest):
    rows = read_tsv(manifest)
    if len(rows) < 2:
        raise RuntimeError("explicit multi-relay manifest must contain at least two routes")
    fields = list(rows[0])
    case_dir = run_dir / "manifests"
    case_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        write_tsv(case_dir / f"relay_{row['relay_phy']}.tsv", [row], fields)
    write_tsv(case_dir / "all.tsv", rows, fields)
    write_tsv(case_dir / "all_reversed.tsv", list(reversed(rows)), fields)
    (run_dir / "manifest_path.txt").write_text(str((case_dir / "all.tsv").resolve()) + "\n")
    print(case_dir)


def load_timings(case_dir):
    samples = {}
    failures = []
    for log in sorted(case_dir.glob("*.log")):
        text = log.read_text(errors="replace")
        match = RESULT_RE.search(text)
        if not match:
            failures.append(log.stem)
            continue
        result = json.loads(match.group(1))
        base = re.sub(r"_r\d+$", "", log.stem)
        samples.setdefault(base, []).append(float(result["host_batch_avg_us"]))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return samples, medians, failures


def load_counters(run_dir):
    candidates = sorted((run_dir / "footprint").rglob("hccn_counter_deltas.tsv"))
    if not candidates:
        return {}, None
    path = candidates[-1]
    counters = {}
    for row in read_tsv(path):
        key = (int(row["physical_device"]), int(row["udie"]),
               int(row["port"]), row["counter"])
        counters[key] = int(row["delta"])
    return counters, path


def counter(counters, device, die, port, name):
    return counters.get((int(device), int(die), int(port), name), 0)


def chain_error(values):
    top = max(values) if values else 0
    return 1.0 if top <= 0 else (top - min(values)) / top


def analyze(run_dir):
    manifest = run_dir / "manifests" / "all.tsv"
    routes = read_tsv(manifest)
    samples, medians, missing_logs = load_timings(run_dir / "cases")
    counters, counter_path = load_counters(run_dir)
    route_rows = []
    for route in routes:
        src = int(route["src_phy"])
        dst = int(route["dst_phy"])
        relay = int(route["relay_phy"])
        src_die = int(route["src_die"])
        dst_die = int(route["dst_die"])
        relay_die = int(route["relay_die"])
        src_port = int(route["src_port"])
        dst_port = int(route["dst_port"])
        ingress = int(route["relay_port_from_src"])
        egress = int(route["relay_port_to_dst"])
        forward = [
            counter(counters, src, src_die, src_port, "tx_busi_flit_num"),
            counter(counters, relay, relay_die, ingress, "rx_busi_flit_num"),
            counter(counters, relay, relay_die, egress, "tx_busi_flit_num"),
            counter(counters, dst, dst_die, dst_port, "rx_busi_flit_num"),
        ]
        reverse = [
            counter(counters, dst, dst_die, dst_port, "tx_busi_flit_num"),
            counter(counters, relay, relay_die, egress, "rx_busi_flit_num"),
            counter(counters, relay, relay_die, ingress, "tx_busi_flit_num"),
            counter(counters, src, src_die, src_port, "rx_busi_flit_num"),
        ]
        fwd_error = chain_error(forward)
        rev_error = chain_error(reverse)
        route_rows.append({
            "relay_phy": relay,
            "weight": int(route["weight"]),
            "forward_chain": ",".join(map(str, forward)),
            "reverse_chain": ",".join(map(str, reverse)),
            "forward_balance_error": f"{fwd_error:.6f}",
            "reverse_balance_error": f"{rev_error:.6f}",
            "active": "YES" if min(forward + reverse) > 1000 else "NO",
            "balanced": "YES" if fwd_error <= 0.10 and rev_error <= 0.10 else "NO",
        })
    write_tsv(run_dir / "explicit_relay_path_validation.tsv", route_rows,
              list(route_rows[0]) if route_rows else [])

    serial = medians.get("serial")
    concurrent = medians.get("concurrent")
    concurrent_reverse = medians.get("concurrent_reverse")
    correct_cases = not missing_logs and all(
        name in medians for name in ("serial", "concurrent", "concurrent_reverse")
    ) and all(f"single_relay{row['relay_phy']}" in medians for row in routes)
    paths_confirmed = bool(counters) and all(
        row["active"] == "YES" and row["balanced"] == "YES" for row in route_rows
    )
    faster_than_serial = serial is not None and concurrent is not None and concurrent < serial
    order_stable = (concurrent is not None and concurrent_reverse is not None and
                    abs(concurrent - concurrent_reverse) / max(concurrent, concurrent_reverse) <= 0.20)
    conclusion = (
        "EXPLICIT_MULTIRELAY_CONFIRMED" if correct_cases and paths_confirmed
        else "FUNCTIONAL_ONLY" if correct_cases
        else "INCONCLUSIVE"
    )
    summary = {
        "conclusion": conclusion,
        "relay_cards": [int(row["relay_phy"]) for row in routes],
        "timing_median_us": medians,
        "correctness_cases_complete": correct_cases,
        "hccn_paths_confirmed": paths_confirmed,
        "concurrent_faster_than_serial": faster_than_serial,
        "concurrent_order_stable": order_stable,
        "missing_result_logs": missing_logs,
        "hccn_counter_file": str(counter_path) if counter_path else "",
    }
    (run_dir / "explicit_multirelay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# A5 CCU explicit multi-relay report", "",
        f"- conclusion: **{conclusion}**",
        f"- explicitly requested relays: `{summary['relay_cards']}`",
        f"- all correctness/timing cases complete: `{correct_cases}`",
        f"- every relay's forward and reverse HCCN chain confirmed: `{paths_confirmed}`",
        f"- concurrent faster than serial: `{faster_than_serial}`",
        f"- reversed channel order within 20%: `{order_stable}`", "",
        "## Timing medians (host batch)", "",
        "| case | median_us | samples |", "|---|---:|---:|",
    ]
    for name in sorted(medians):
        lines.append(f"| {name} | {medians[name]:.3f} | {len(samples[name])} |")
    lines += ["", "## Explicit physical relay chains", "",
              "Each chain contains four HCCN flit deltas in wire order. A path is balanced when "
              "the max/min spread is at most 10%.", "",
              "| relay | weight | forward chain | reverse chain | active | balanced |",
              "|---:|---:|---|---|---|---|"]
    for row in route_rows:
        lines.append(
            f"| {row['relay_phy']} | {row['weight']} | `{row['forward_chain']}` | "
            f"`{row['reverse_chain']}` | {row['active']} | {row['balanced']} |"
        )
    lines += ["", "## Boundary", "",
              "HCCN before/after deltas prove that both named physical relay chains carried data in "
              "the same workload window. They do not alone prove cycle-level overlap. The CCU "
              "concurrent kernel supplies that causal condition by issuing every channel's WriteNb "
              "before waiting for any completion; the serial control waits after each WriteNb.", ""]
    (run_dir / "explicit_multirelay_report.md").write_text("\n".join(lines))
    print(run_dir / "explicit_multirelay_report.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.manifest is None:
            parser.error("--prepare requires --manifest")
        prepare(args.run_dir, args.manifest)
    else:
        analyze(args.run_dir)


if __name__ == "__main__":
    main()
