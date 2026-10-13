#!/usr/bin/env python3
"""Compare validated protocols; hardware Ccu time is separate from host batch time."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from analyze_a5_ccu_peer_plan import analyze as analyze_peer
from analyze_a5_ccu_local_copy_perf import parse_device

MODES = ("baseline", "packed", "packed-ready")


def analyze(root):
    index = root / "runs.tsv"
    entries = []
    if index.is_file():
        with index.open() as file:
            entries = list(csv.DictReader(file, delimiter="\t"))
    seen = set()
    rows = []
    complete = len(entries) == len(MODES)
    profile_complete = complete
    comparison_key = None
    configuration_consistent = True
    for entry in entries:
        mode = entry["mode"]
        if mode not in MODES or mode in seen:
            raise ValueError("invalid/duplicate mode in runs.tsv")
        seen.add(mode)
        run = Path(entry["run_dir"]) if entry.get("run_dir") else None
        functional = False
        if run and (run / "plans/peer_plan.json").is_file() and (run / "run_settings.json").is_file():
            peer = analyze_peer(run)
            settings = json.loads((run / "run_settings.json").read_text())
            # Reject accidental cross-run changes in paths, bytes or timing
            # settings. These are not evidence of a notification improvement.
            key = (json.loads((run / "plans/peer_plan.json").read_text()),
                   {name: settings.get(name) for name in ("bytes_per_peer", "dtype", "warmup",
                    "iterations", "repeats", "sync_mode", "graph_backend", "self_copy_mode")})
            if comparison_key is None:
                comparison_key = key
            else:
                configuration_consistent &= key == comparison_key
            functional = (entry["status"] == "0" and peer["complete"] and
                          settings.get("channel_sync_mode") == mode)
            k = len(peer["devices"])
            for repeat in range(1, settings["repeats"] + 1):
                for rank in range(k):
                    result_file = run / f"r{repeat}/results/rank{rank}.json"
                    try:
                        result = json.loads(result_file.read_text()) if result_file.is_file() else {}
                        if not isinstance(result, dict):
                            result = {}
                    except (OSError, ValueError):
                        result = {}
                    events, copies, sources = parse_device(run / f"r{repeat}/profiling/rank{rank}", result.get("pid"))
                    accepted = functional and len(events) == settings["iterations"] and copies == 0
                    durations = [e["duration_us"] for e in events]
                    rows.append(dict(mode=mode, repeat=repeat, rank=rank, functional=functional,
                        device_samples=len(events), expected_samples=settings["iterations"],
                        device_accepted=accepted, memcpy_samples=copies,
                        min_us=min(durations) if accepted else None,
                        median_us=statistics.median(durations) if accepted else None,
                        mean_us=statistics.mean(durations) if accepted else None,
                        max_us=max(durations) if accepted else None,
                        host_avg_us=result.get("host_avg_us"), sources=sources))
                    profile_complete &= accepted
        else:
            rows.append(dict(mode=mode, functional=False, error="missing result directory"))
            profile_complete = False
        complete &= functional
    complete &= seen == set(MODES)
    complete &= configuration_consistent
    profile_complete &= configuration_consistent
    medians = {mode: max(statistics.median([r["median_us"] for r in rows
        if r.get("device_accepted") and r["mode"] == mode and r["rank"] == rank])
        for rank in {r["rank"] for r in rows if r.get("device_accepted") and r["mode"] == mode})
        for mode in MODES if any(r.get("device_accepted") and r["mode"] == mode for r in rows)}
    gains = {mode: medians["baseline"] / medians[mode] for mode in MODES[1:]
             if profile_complete and "baseline" in medians and mode in medians}
    summary = dict(functional_complete=complete, device_profile_complete=profile_complete,
                   configuration_consistent=configuration_consistent,
                   slowest_rank_ccu_median_us=medians, baseline_speedup=gains, rows=rows)
    (root / "channel_sync_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# CCU Channel notification/submission A/B/C", "",
        f"- correctness + lifecycle complete: `{complete}`",
        f"- exact formal Ccu task counts on every rank: `{profile_complete}`", "",
        "| mode | repeat | rank | correctness | Ccu samples | min us | median us | mean us | max us | host avg us |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append("| " + " | ".join(str(row.get(key, "NA")) for key in
            ("mode", "repeat", "rank", "functional", "device_samples", "min_us", "median_us",
             "mean_us", "max_us", "host_avg_us")) + " |")
    report += ["", f"Baseline-relative Ccu speedup (only complete profiles): `{gains}`", "",
        "Hardware Ccu duration includes peer waiting and self copy; host avg includes submission/synchronization.",
        "Missing/extra Ccu tasks, missing ranks, nonzero exit or silent protocol fallback invalidate the comparison.",
        "All protocols preserve per-Channel completion handshakes. packed-ready removes the all-Channel pre-write barrier;",
        "it is still ordered interpretation, not out-of-order scheduling. PASS proves correctness, not a speedup.",
        "Open each indexed run's r*/profiling/rank*/*_ascend_pt in MindStudio. Shared-relay contention can affect results."]
    (root / "channel_sync_report.md").write_text("\n".join(report) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.run_dir)
    print(args.run_dir / "channel_sync_report.md")
