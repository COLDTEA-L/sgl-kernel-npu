#!/usr/bin/env python3
"""Validate native baseline per-rank files, including interrupted repeats."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def valid_result(result, rank, settings):
    def number(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0)
    k, count = len(settings["devices"]), settings["iterations"]
    samples = result.get("host_call_us")
    mode = result.get("sync_mode", "per-call")
    if mode != settings.get("sync_mode", "per-call"):
        return False
    if mode == "batch":
        timing_valid = (samples == [] and number(result.get("host_batch_total_us"))
            and number(result.get("host_avg_us"))
            and math.isclose(result["host_avg_us"], result["host_batch_total_us"] / count,
                             rel_tol=1e-6, abs_tol=1e-6))
    elif mode == "per-call":
        timing_valid = (isinstance(samples, list) and len(samples) == count
            and all(number(x) for x in samples) and number(result.get("host_avg_us"))
            and math.isclose(result["host_avg_us"], sum(samples) / count,
                             rel_tol=1e-6, abs_tol=1e-6))
    else:
        return False
    return (type(result.get("rank")) is int and result["rank"] == rank
        and type(result.get("ranks")) is int and result["ranks"] == k
        and result.get("implementation") == "native_hccl"
        and result.get("api") == "dist.all_to_all_single"
        and result.get("correctness") == "PASS"
        and result.get("bytes_per_peer") == settings["bytes_per_peer"]
        and result.get("tensor_bytes") == k * settings["bytes_per_peer"]
        and result.get("warmup") == settings["warmup"]
        and result.get("iterations") == count and timing_valid)


def analyze(root):
    settings = json.loads((root / "run_settings.json").read_text())
    k = len(settings["devices"])
    rows = []
    for repeat in range(1, settings["repeats"] + 1):
        case = f"native_alltoall_r{repeat}"
        status_file = root / "cases" / f"{case}.status"
        status = status_file.read_text().strip() if status_file.is_file() else "INTERRUPTED_OR_NOT_RUN"
        results, invalid = [], []
        for rank in range(k):
            path = root / f"r{repeat}" / "results" / f"rank{rank}.json"
            try:
                result = json.loads(path.read_text())
                if not isinstance(result, dict) or not valid_result(result, rank, settings):
                    raise ValueError("invalid result fields")
                results.append(result)
            except (OSError, ValueError, TypeError):
                invalid.append(rank)
        data_valid = not invalid
        valid = data_valid and status == "0"
        rows.append(dict(case=case, status=status,
            correctness="PASS" if data_valid else "INCOMPLETE_OR_FAIL",
            lifecycle="PASS" if valid else "FAILED_OR_INTERRUPTED_AFTER_DATA_PASS" if data_valid else "INCOMPLETE_OR_FAIL",
            invalid_or_missing_ranks=','.join(map(str, invalid)),
            slowest_rank_host_avg_us=max(r["host_avg_us"] for r in results) if data_valid else None))
    complete = bool(rows) and all(row["lifecycle"] == "PASS" for row in rows)
    valid_times = [row["slowest_rank_host_avg_us"] for row in rows if row["lifecycle"] == "PASS"]
    summary = dict(complete=complete, implementation="native_hccl", devices=settings["devices"],
        sync_mode=settings.get("sync_mode", "per-call"),
        bytes_per_peer=settings["bytes_per_peer"], tensor_bytes=k * settings["bytes_per_peer"],
        network_send_bytes=(k - 1) * settings["bytes_per_peer"], cases=rows,
        median_slowest_rank_host_us=statistics.median(valid_times) if valid_times else None)
    with (root / "native_alltoall_results.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    (root / "native_alltoall_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# Native HCCL AllToAll baseline", "", f"- complete: `{complete}`",
        "- API: `dist.all_to_all_single(recv, send)`; no custom plan/Channel/relay selection.",
        f"- physical communicator cards: `{settings['devices']}`",
        f"- per peer: `{settings['bytes_per_peer']}` bytes; tensor: `{summary['tensor_bytes']}` bytes per rank.",
        f"- network send payload: `{summary['network_send_bytes']}` bytes per rank, excluding self.",
        f"- warmup: `{settings['warmup']}`; measured calls: `{settings['iterations']}`; graph: `none`.",
        f"- synchronization mode: `{summary['sync_mode']}`; batch means one final device synchronization.",
        "", "| case | status | correctness | lifecycle | invalid/missing ranks | slowest-rank host us |",
        "|---|---|---|---|---|---:|"]
    for row in rows:
        report.append(f"| {row['case']} | {row['status']} | {row['correctness']} | {row['lifecycle']} | {row['invalid_or_missing_ranks']} | {row['slowest_rank_host_avg_us']} |")
    report.extend(["", f"Median across complete repeats: `{summary['median_slowest_rank_host_us']}` us.",
        "", "## Interpretation", "",
        "Host timing includes submission and synchronization, not initialization, first-call resource creation or warmup.",
        "In batch mode, host us is total batch time / call count, not an individual-call latency sample. Inspect CCU tasks for individual device durations.",
        "CCU_SCHED requests the native CCU scheduler; check profiling to confirm the actual device implementation.",
        "Native HCCL chooses its own paths. This baseline is not a forced direct-only path, candidate0 or candidate2.",
        "Use the same cards, per-peer bytes, warmup/iterations, synchronization mode and profiling setting for comparisons. Peer-plan currently completes each invocation; do not silently remove that safety boundary.",
        "Host call timing is not CCU task timing or link bandwidth; use MindStudio device tasks for that comparison.",
        "Only all-rank correctness PASS and exit status 0 qualify as a complete repeat.",
        "Rank-local atomic JSON files are authoritative; interleaved terminal output is not parsed as data.",
        "", "## Profiling", "",
        "With --profile: r*/profiling/rank*/ contains the exported *_ascend_pt directories.",
        "Open the per-rank export in MindStudio; warmup is outside collection."])
    (root / "native_alltoall_report.md").write_text("\n".join(report) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_dir)
    print(args.run_dir / "native_alltoall_report.md")
