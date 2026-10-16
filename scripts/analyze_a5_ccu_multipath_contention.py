#!/usr/bin/env python3
"""Analyze two prepared multi-path jobs sharing directed physical links."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

support_dir = Path(__file__).resolve().parents[1] / "tests/python/deepep"
sys.path.insert(0, str(support_dir))
from a5_ccu_path_contention_support import format_size_label, overlap_ns  # noqa: E402


def read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5)))
    return ordered[index]


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def trace_size_root(run_dir, size):
    """Use human-readable paths for new runs and accept numeric legacy runs."""
    current = run_dir / "traces" / f"size_{format_size_label(size)}"
    legacy = run_dir / "traces" / f"size_{size}"
    return current if current.exists() or not legacy.exists() else legacy


def concurrent_trace_dir(repeat_root, job):
    current = repeat_root / "concurrent_PQ" / job
    legacy = repeat_root / f"concurrent_{job}"
    return current if current.exists() or not legacy.exists() else legacy


def load_collective_batch(directory, expected_iterations):
    documents = []
    for rank in (0, 1):
        path = directory / f"rank{rank}.json"
        if not path.is_file():
            raise RuntimeError(f"missing batch trace: {path}")
        document = json.loads(path.read_text())
        records = document.get("records", [])
        batch_only = document.get("iteration_timing_semantics") == "batch_only"
        if batch_only and (records or document.get("iterations") != expected_iterations or
                           document.get("loop_body") != "launch_only"):
            raise RuntimeError(f"invalid launch-only batch metadata: {path}")
        if not batch_only and len(records) != expected_iterations:
            raise RuntimeError(
                f"{path} has {len(records)} submissions, expected {expected_iterations}"
            )
        if document.get("completion_mode") != "batch":
            raise RuntimeError(f"{path} is not a batch-completion trace")
        if not batch_only and document.get("iteration_timing_semantics") != "host_submission_only":
            raise RuntimeError(f"{path} has invalid batch timing semantics")
        for offset, record in enumerate(records, 1):
            if record.get("iteration") != offset:
                raise RuntimeError(f"submission numbering mismatch in {path}")
        documents.append(document)
    started = min(int(item["batch_started_ns"]) for item in documents)
    ended = max(int(item["batch_ended_ns"]) for item in documents)
    if ended <= started:
        raise RuntimeError(f"non-positive batch interval in {directory}")
    return {
        "started_ns": started,
        "ended_ns": ended,
        "batch_window_us": (ended - started) / 1000.0,
        "batch_average_us": max(float(item["batch_average_us"]) for item in documents),
        "documents": documents,
    }


def render_batch_report(settings, shared_directions, link_identity, aggregate):
    lines = [
        "# A5 multi-path physical-direction contention report", "",
        "- complete: `True`",
        f"- shared directed physical hops: `{shared_directions}`",
        f"- shared physical link identity: `{link_identity}`",
        f"- sizes (bytes per peer slice): `{settings['sizes']}`",
        f"- warmup / measured calls / repeats: `{settings['warmup']} / {settings['iterations']} / {settings['repeats']}`",
        "- P and Q align only their first measured submission; each instance then enqueues the full loop independently.",
        "- each rank calls torch.npu.synchronize() once after the measured loop, never inside it.",
        "- host submission records are not device execution intervals and are not used to infer per-iteration conflicts.",
        "- conflict probability must be counted from the concurrent P/Q MindStudio device timelines.",
        "", "## Batch slowdown summary", "",
        "| bytes/peer | job | baseline avg us | concurrent avg us | inflation median | inflation p95 | inflation max | host batch overlap | conflict probability |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['bytes_per_peer']} | {row['job']} | "
            f"{fmt(row['baseline_batch_average_us_median'])} | "
            f"{fmt(row['concurrent_batch_average_us_median'])} | "
            f"{fmt(row['time_inflation_median'])}x | "
            f"{fmt(row['time_inflation_p95'])}x | "
            f"{fmt(row['time_inflation_max'])}x | "
            f"{fmt(row['host_batch_overlap_fraction_median'])} | MindStudio timeline required |"
        )
    lines += [
        "", "## MindStudio counting rule", "",
        "For each job, count a measured device invocation as conflicted only when its CCU/communication task interval "
        "overlaps a measured invocation of the other job on the shared physical direction. Divide that count by the "
        "measured invocation count. New runs contain no per-iteration Python markers or timers; "
        "number the measured CCU launches in order on each rank's execution stream. "
        "Legacy host markers, when present, are not device intervals.", "",
    ]
    return "\n".join(lines)


def analyze_batch(run_dir, settings):
    sizes = [int(item) for item in settings["sizes"]]
    repeats = int(settings["repeats"])
    iterations = int(settings["iterations"])
    p_rows = read_tsv(run_dir / "manifests/P.tsv")
    q_rows = read_tsv(run_dir / "manifests/Q.tsv")
    if len(p_rows) != 1 or len(q_rows) != 1:
        raise RuntimeError("the contention experiment requires exactly one relay per job")
    link_identity = shared_link_identity(p_rows[0], q_rows[0])
    shared = sorted(route_directions(p_rows[0]) & route_directions(q_rows[0]))
    if not shared:
        raise RuntimeError("P and Q manifests do not share a directed physical hop")
    shared_directions = [f"{src}->{dst}" for src, dst in shared]

    batch_rows = []
    submission_rows = []
    baselines = {}
    for size in sizes:
        size_root = trace_size_root(run_dir, size)
        for job in ("P", "Q"):
            samples = []
            for repeat in range(1, repeats + 1):
                batch = load_collective_batch(
                    size_root / f"repeat_{repeat}" / f"baseline_{job}", iterations
                )
                samples.append(batch["batch_average_us"])
                batch_rows.append({
                    "bytes_per_peer": size, "repeat": repeat, "phase": "baseline",
                    "job": job, "iterations": iterations,
                    "batch_started_ns": batch["started_ns"],
                    "batch_ended_ns": batch["ended_ns"],
                    "batch_window_us": batch["batch_window_us"],
                    "batch_average_us": batch["batch_average_us"],
                })
                append_submission_rows(
                    submission_rows, settings, size, repeat, "baseline", job, batch
                )
            baselines[(size, job)] = statistics.median(samples)

    concurrent_rows = []
    for size in sizes:
        size_root = trace_size_root(run_dir, size)
        for repeat in range(1, repeats + 1):
            root = size_root / f"repeat_{repeat}"
            batches = {
                job: load_collective_batch(concurrent_trace_dir(root, job), iterations)
                for job in ("P", "Q")
            }
            overlap = overlap_ns(
                (batches["P"]["started_ns"], batches["P"]["ended_ns"]),
                (batches["Q"]["started_ns"], batches["Q"]["ended_ns"]),
            )
            shorter = min(
                batches["P"]["ended_ns"] - batches["P"]["started_ns"],
                batches["Q"]["ended_ns"] - batches["Q"]["started_ns"],
            )
            for job in ("P", "Q"):
                batch = batches[job]
                baseline = baselines[(size, job)]
                inflation = batch["batch_average_us"] / baseline
                row = {
                    "bytes_per_peer": size, "repeat": repeat, "phase": "concurrent_PQ",
                    "job": job, "iterations": iterations,
                    "batch_started_ns": batch["started_ns"],
                    "batch_ended_ns": batch["ended_ns"],
                    "batch_window_us": batch["batch_window_us"],
                    "batch_average_us": batch["batch_average_us"],
                    "baseline_median_us": baseline,
                    "time_inflation": inflation,
                    "time_inflation_percent": (inflation - 1.0) * 100.0,
                    "host_batch_overlap_us": overlap / 1000.0,
                    "host_batch_overlap_fraction": overlap / shorter if shorter else 0.0,
                    "conflict_probability": "MINDSTUDIO_TIMELINE_REQUIRED",
                }
                batch_rows.append(row)
                concurrent_rows.append(row)
                append_submission_rows(
                    submission_rows, settings, size, repeat, "concurrent_PQ", job, batch
                )

    groups = {}
    for row in concurrent_rows:
        groups.setdefault((row["bytes_per_peer"], row["job"]), []).append(row)
    aggregate = []
    for (size, job), rows in sorted(groups.items()):
        inflations = [row["time_inflation"] for row in rows]
        aggregate.append({
            "bytes_per_peer": size,
            "job": job,
            "repeats": len(rows),
            "iterations_per_repeat": iterations,
            "baseline_batch_average_us_median": baselines[(size, job)],
            "concurrent_batch_average_us_median": statistics.median(
                [row["batch_average_us"] for row in rows]
            ),
            "time_inflation_median": statistics.median(inflations),
            "time_inflation_p95": percentile(inflations, 0.95),
            "time_inflation_max": max(inflations),
            "host_batch_overlap_fraction_median": statistics.median(
                [row["host_batch_overlap_fraction"] for row in rows]
            ),
            "conflict_probability": "MINDSTUDIO_TIMELINE_REQUIRED",
        })

    batch_fields = (
        "bytes_per_peer", "repeat", "phase", "job", "iterations",
        "batch_started_ns", "batch_ended_ns", "batch_window_us", "batch_average_us",
        "baseline_median_us", "time_inflation", "time_inflation_percent",
        "host_batch_overlap_us", "host_batch_overlap_fraction", "conflict_probability",
    )
    submission_fields = (
        "bytes_per_peer", "repeat", "phase", "job", "rank", "physical_device",
        "iteration", "enqueue_started_ns", "enqueue_ended_ns", "enqueue_duration_us",
        "timing_semantics",
    )
    aggregate_fields = (
        "bytes_per_peer", "job", "repeats", "iterations_per_repeat",
        "baseline_batch_average_us_median", "concurrent_batch_average_us_median",
        "time_inflation_median", "time_inflation_p95", "time_inflation_max",
        "host_batch_overlap_fraction_median", "conflict_probability",
    )
    write_csv(run_dir / "path_contention_batches.csv", batch_rows, batch_fields)
    write_csv(run_dir / "path_contention_submissions.csv", submission_rows, submission_fields)
    write_csv(run_dir / "path_contention_summary.csv", aggregate, aggregate_fields)

    summary = {
        "complete": True,
        "measurement_mode": "independent_batch_enqueue_then_local_sync",
        "conflict_probability_source": "MindStudio device timeline",
        "host_submission_intervals_are_device_intervals": False,
        "shared_physical_directions": shared_directions,
        "shared_physical_link": link_identity,
        "settings": settings,
        "aggregate": aggregate,
        "per_size_results": {
            str(size): f"results/size_{format_size_label(size)}" for size in sizes
        },
    }
    (run_dir / "path_contention_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    report = render_batch_report(settings, shared_directions, link_identity, aggregate)
    (run_dir / "path_contention_report.md").write_text(report)

    for size in sizes:
        result_dir = run_dir / "results" / f"size_{format_size_label(size)}"
        size_batches = [row for row in batch_rows if row["bytes_per_peer"] == size]
        size_submissions = [row for row in submission_rows if row["bytes_per_peer"] == size]
        size_aggregate = [row for row in aggregate if row["bytes_per_peer"] == size]
        write_csv(result_dir / "path_contention_batches.csv", size_batches, batch_fields)
        write_csv(result_dir / "path_contention_submissions.csv", size_submissions,
                  submission_fields)
        write_csv(result_dir / "path_contention_summary.csv", size_aggregate,
                  aggregate_fields)
        size_summary = dict(summary)
        size_summary["bytes_per_peer"] = size
        size_summary["settings"] = dict(settings, sizes=[size])
        size_summary["aggregate"] = size_aggregate
        size_summary["per_size_results"] = {str(size): f"results/size_{format_size_label(size)}"}
        (result_dir / "path_contention_summary.json").write_text(
            json.dumps(size_summary, indent=2, sort_keys=True) + "\n"
        )
        (result_dir / "path_contention_report.md").write_text(
            render_batch_report(size_summary["settings"], shared_directions,
                                link_identity, size_aggregate)
        )
    print(json.dumps({"complete": True, "aggregate_rows": len(aggregate),
                      "conflict_probability_source": "MindStudio device timeline"}))


def append_submission_rows(output, settings, size, repeat, phase, job, batch):
    endpoints = settings["jobs"][job]["endpoints"]
    for document in batch["documents"]:
        rank = int(document["rank"])
        for record in document["records"]:
            output.append({
                "bytes_per_peer": size,
                "repeat": repeat,
                "phase": phase,
                "job": job,
                "rank": rank,
                "physical_device": endpoints[rank],
                "iteration": record["iteration"],
                "enqueue_started_ns": record["started_ns"],
                "enqueue_ended_ns": record["ended_ns"],
                "enqueue_duration_us": record["duration_us"],
                "timing_semantics": record.get("timing_semantics", "host_submission_only"),
            })


def analyze_baseline_resources(run_dir, settings):
    """Compare identical launches with and without the other job resident."""
    from analyze_a5_ccu_local_copy_perf import parse_device

    iterations = int(settings["iterations"])
    profiled = settings.get("mindstudio_profile", False)
    cases = []
    device_complete = bool(profiled)
    for size in settings["sizes"]:
        for repeat in range(1, int(settings["repeats"]) + 1):
            for job in ("P", "Q"):
                for mode, phase in (("solo", f"solo_baseline_{job}"),
                                    ("resident", f"baseline_{job}")):
                    root = trace_size_root(run_dir, size) / f"repeat_{repeat}"
                    batch = load_collective_batch(root / phase, iterations)
                    for document in batch["documents"]:
                        rank = int(document["rank"])
                        profile_root = (run_dir / "profiles" / f"size_{format_size_label(size)}" /
                                        f"repeat_{repeat}" / phase / f"rank{rank}")
                        events, copies, sources = parse_device(profile_root, document.get("pid")) if profiled else ([], 0, [])
                        accepted = profiled and len(events) == iterations and copies == 0
                        device_complete = device_complete and accepted
                        values = [event["duration_us"] for event in events] if accepted else []
                        cases.append(dict(
                            bytes_per_peer=size, size_label=format_size_label(size),
                            repeat=repeat, job=job, rank=rank,
                            physical_device=settings["jobs"][job]["endpoints"][rank],
                            mode=mode, host_batch_average_us=float(document["batch_average_us"]),
                            device_samples=len(events), expected_samples=iterations,
                            memcpy_samples=copies, device_accepted=accepted,
                            ccu_min_us=min(values) if values else None,
                            ccu_median_us=statistics.median(values) if values else None,
                            ccu_max_us=max(values) if values else None,
                            sources=sources))
    rows = []
    for size in settings["sizes"]:
        for job in ("P", "Q"):
            for rank in (0, 1):
                groups = {mode: [item for item in cases if item["bytes_per_peer"] == size
                                and item["job"] == job and item["rank"] == rank and item["mode"] == mode]
                          for mode in ("solo", "resident")}
                row = dict(bytes_per_peer=size, size_label=format_size_label(size), job=job,
                           rank=rank, physical_device=settings["jobs"][job]["endpoints"][rank])
                for mode, group in groups.items():
                    row[f"{mode}_host_average_us"] = statistics.median(
                        [item["host_batch_average_us"] for item in group])
                    accepted = all(item["device_accepted"] for item in group)
                    for metric in ("min", "median", "max"):
                        row[f"{mode}_ccu_{metric}_us"] = statistics.median(
                            [item[f"ccu_{metric}_us"] for item in group]) if accepted else None
                row["resident_over_solo_host_ratio"] = row["resident_host_average_us"] / row["solo_host_average_us"]
                row["resident_over_solo_ccu_ratio"] = (
                    row["resident_ccu_median_us"] / row["solo_ccu_median_us"]
                    if row["solo_ccu_median_us"] is not None and row["resident_ccu_median_us"] is not None else None)
                rows.append(row)

    complete = not profiled or device_complete
    summary = dict(complete=complete, device_profile_complete=device_complete,
                   experiment_mode="baseline-check", settings=settings, aggregate=rows,
                   cases=cases, formal_loop_body="launch_only")

    def save(root, size_rows, size_cases, current_settings):
        root.mkdir(parents=True, exist_ok=True)
        write_csv(root / "baseline_resource_summary.csv", size_rows, tuple(rows[0]))
        write_csv(root / "baseline_resource_cases.csv", size_cases, tuple(cases[0]))
        current = dict(summary, aggregate=size_rows, cases=size_cases, settings=current_settings)
        (root / "baseline_resource_summary.json").write_text(json.dumps(current, indent=2) + "\n")
        lines = ["# Baseline resource residency comparison", "",
                 f"- complete: `{complete}`; device profile complete: `{device_complete}`",
                 "- solo: only the measured job has endpoint processes, communicators and plans.",
                 "- resident: both jobs have resources; only the measured job submits work.",
                 "- paths, weights, dtype, HCCL buffer budget, warmup and formal loop are identical.",
                 "- no concurrent phase; warmup/queued correctness checks are outside profiling.",
                 "- host batch average includes submission gaps and final synchronization; compare CCU separately.",
                 "- ratios describe the whole idle-instance environment (processes + runtime + plans), not a single resource type.", "",
                 "| size | job | rank/card | solo host us | resident host us | solo CCU median us | resident CCU median us | CCU ratio |",
                 "|---|---|---|---:|---:|---:|---:|---:|"]
        for row in size_rows:
            lines.append(f"| {row['size_label']} | {row['job']} | {row['rank']}/{row['physical_device']} | "
                         f"{fmt(row['solo_host_average_us'])} | {fmt(row['resident_host_average_us'])} | "
                         f"{fmt(row['solo_ccu_median_us'])} | {fmt(row['resident_ccu_median_us'])} | "
                         f"{fmt(row['resident_over_solo_ccu_ratio'])} |")
        lines += ["", "Missing/extra device tasks are marked incomplete, never replaced by host timing.",
                  "Inspect both rank profiles and launch gaps; sequential solo/resident runs may also reflect other users' activity.", ""]
        report = "\n".join(lines)
        (root / "baseline_resource_report.md").write_text(report)
        (root / "path_contention_report.md").write_text(report)

    save(run_dir, rows, cases, settings)
    for size in settings["sizes"]:
        save(run_dir / "results" / f"size_{format_size_label(size)}",
             [row for row in rows if row["bytes_per_peer"] == size],
             [row for row in cases if row["bytes_per_peer"] == size], dict(settings, sizes=[size]))
    print(json.dumps(dict(complete=complete, experiment_mode="baseline-check",
                          device_profile_complete=device_complete, aggregate_rows=len(rows))))
    return summary


def render_report(sizes, repeats, iterations, shared_directions, link_identity, aggregate):
    lines = [
        "# A5 multi-path physical-direction contention report", "",
        "- complete: `True`",
        f"- shared directed physical hops: `{shared_directions}`",
        f"- shared physical link identity: `{link_identity}`",
        f"- sizes (bytes per peer slice): `{sizes}`",
        f"- repeats / measured iterations: `{repeats} / {iterations}`",
        "- warmup is completed before profiling and is absent from the iteration tables.",
        "- one conflict pair means one P invocation and one Q invocation had a non-empty host-monotonic execution overlap.",
        "- timing inflation is concurrent invocation time divided by the same job/size solo baseline median.",
        "", "## Conflict summary", "",
        "| bytes/peer | job | conflicts/iters | overlap pairs | baseline us | conflict median us | inflation median | inflation p95 | inflation max | median overlap |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['bytes_per_peer']} | {row['job']} | "
            f"{row['conflict_iterations']}/{row['iterations']} | {row['overlapping_pairs']} | "
            f"{fmt(row['baseline_median_us'])} | {fmt(row['conflict_duration_median_us'])} | "
            f"{fmt(row['conflict_inflation_median'])}x | {fmt(row['conflict_inflation_p95'])}x | "
            f"{fmt(row['conflict_inflation_max'])}x | {fmt(row['overlap_fraction_median'])} |"
        )
    lines += [
        "", "## Interpretation boundary", "",
        "The manifests prove that every invocation has a positive-weight relay path over the shared directed hops. "
        "Interval overlap therefore identifies an attempted collision window, while slowdown relative to the solo "
        "baseline quantifies its observed impact. Host intervals do not prove cycle-level arbitration on a port; "
        "use the optional MindStudio traces and HCCN counters for that final attribution.", "",
    ]
    return "\n".join(lines)


def route_directions(row):
    src = int(row["src_phy"])
    dst = int(row["dst_phy"])
    relay = int(row["relay_phy"])
    return {(src, relay), (relay, dst), (dst, relay), (relay, src)}


def shared_link_identity(p_row, q_row):
    """Require P's source--relay edge to be Q's source--relay edge in reverse.

    EIDs may differ between jobs.  Physical card, UDie and port identities may not.
    """
    matches = (
        int(p_row["src_phy"]) == int(q_row["relay_phy"])
        and int(p_row["relay_phy"]) == int(q_row["src_phy"])
        and int(p_row["src_die"]) == int(q_row["relay_die"])
        and int(p_row["src_port"]) == int(q_row["relay_port_from_src"])
        and int(p_row["relay_die"]) == int(q_row["src_die"])
        and int(p_row["relay_port_from_src"]) == int(q_row["src_port"])
        and p_row["src_edge"] == q_row["src_edge"]
    )
    if not matches:
        p_src = int(p_row["src_phy"])
        p_relay = int(p_row["relay_phy"])
        raise RuntimeError(
            f"P and Q resolve card {p_src}<->{p_relay} through different physical UDie/ports; "
            "select relay planes that identify one shared physical edge"
        )
    return {
        "p_source_phy": int(p_row["src_phy"]),
        "p_source_udie": int(p_row["src_die"]),
        "p_source_port": int(p_row["src_port"]),
        "p_relay_phy": int(p_row["relay_phy"]),
        "p_relay_udie": int(p_row["relay_die"]),
        "p_relay_port": int(p_row["relay_port_from_src"]),
        "topology_edge": p_row["src_edge"],
    }


def load_collective_intervals(directory, expected_iterations):
    documents = []
    for rank in (0, 1):
        path = directory / f"rank{rank}.json"
        if not path.is_file():
            raise RuntimeError(f"missing iteration trace: {path}")
        document = json.loads(path.read_text())
        records = document.get("records", [])
        if len(records) != expected_iterations:
            raise RuntimeError(
                f"{path} has {len(records)} iterations, expected {expected_iterations}"
            )
        documents.append(document)
    combined = []
    for offset in range(expected_iterations):
        pair = [document["records"][offset] for document in documents]
        iteration = offset + 1
        if any(item.get("iteration") != iteration for item in pair):
            raise RuntimeError(f"iteration numbering mismatch in {directory}")
        started = min(int(item["started_ns"]) for item in pair)
        ended = max(int(item["ended_ns"]) for item in pair)
        if ended <= started:
            raise RuntimeError(f"non-positive collective interval in {directory}")
        combined.append({
            "iteration": iteration,
            "started_ns": started,
            "ended_ns": ended,
            "duration_us": max(float(item["duration_us"]) for item in pair),
            "collective_window_us": (ended - started) / 1000.0,
        })
    return combined


def analyze(run_dir):
    settings = json.loads((run_dir / "run_settings.json").read_text())
    if settings.get("experiment_mode") == "baseline-check":
        return analyze_baseline_resources(run_dir, settings)
    if settings.get("completion_mode") == "batch":
        return analyze_batch(run_dir, settings)
    sizes = [int(item) for item in settings["sizes"]]
    repeats = int(settings["repeats"])
    iterations = int(settings["iterations"])
    p_rows = read_tsv(run_dir / "manifests/P.tsv")
    q_rows = read_tsv(run_dir / "manifests/Q.tsv")
    if len(p_rows) != 1 or len(q_rows) != 1:
        raise RuntimeError("the contention experiment requires exactly one relay per job")
    link_identity = shared_link_identity(p_rows[0], q_rows[0])
    shared = sorted(route_directions(p_rows[0]) & route_directions(q_rows[0]))
    if not shared:
        raise RuntimeError("P and Q manifests do not share a directed physical hop")

    baselines = {}
    baseline_rows = []
    for size in sizes:
        for job in ("P", "Q"):
            samples = []
            for repeat in range(1, repeats + 1):
                directory = (trace_size_root(run_dir, size) /
                             f"repeat_{repeat}" / f"baseline_{job}")
                intervals = load_collective_intervals(directory, iterations)
                for item in intervals:
                    samples.append(item["duration_us"])
                    baseline_rows.append({
                        "bytes_per_peer": size,
                        "repeat": repeat,
                        "job": job,
                        "iteration": item["iteration"],
                        "duration_us": item["duration_us"],
                    })
            baselines[(size, job)] = statistics.median(samples)

    rows = []
    pair_counts = {}
    for size in sizes:
        for repeat in range(1, repeats + 1):
            root = trace_size_root(run_dir, size) / f"repeat_{repeat}"
            intervals = {
                job: load_collective_intervals(concurrent_trace_dir(root, job), iterations)
                for job in ("P", "Q")
            }
            overlaps = {"P": {item["iteration"]: [] for item in intervals["P"]},
                        "Q": {item["iteration"]: [] for item in intervals["Q"]}}
            pairs = 0
            for left in intervals["P"]:
                for right in intervals["Q"]:
                    amount = overlap_ns(
                        (left["started_ns"], left["ended_ns"]),
                        (right["started_ns"], right["ended_ns"]),
                    )
                    if amount <= 0:
                        continue
                    pairs += 1
                    overlaps["P"][left["iteration"]].append((right["iteration"], amount))
                    overlaps["Q"][right["iteration"]].append((left["iteration"], amount))
            pair_counts[(size, repeat)] = pairs
            for job in ("P", "Q"):
                baseline = baselines[(size, job)]
                for item in intervals[job]:
                    matches = overlaps[job][item["iteration"]]
                    overlap_total = sum(amount for _, amount in matches)
                    window_ns = item["ended_ns"] - item["started_ns"]
                    inflation = item["duration_us"] / baseline
                    rows.append({
                        "bytes_per_peer": size,
                        "repeat": repeat,
                        "job": job,
                        "iteration": item["iteration"],
                        "started_ns": item["started_ns"],
                        "ended_ns": item["ended_ns"],
                        "duration_us": item["duration_us"],
                        "collective_window_us": item["collective_window_us"],
                        "baseline_median_us": baseline,
                        "conflict": "YES" if matches else "NO",
                        "peer_iterations": ",".join(str(peer) for peer, _ in matches),
                        "overlap_us": overlap_total / 1000.0,
                        "overlap_fraction": min(overlap_total, window_ns) / window_ns,
                        "time_inflation": inflation,
                        "time_inflation_percent": (inflation - 1.0) * 100.0,
                    })

    groups = {}
    for row in rows:
        groups.setdefault((row["bytes_per_peer"], row["job"]), []).append(row)
    aggregate = []
    for (size, job), group in sorted(groups.items()):
        conflicts = [row for row in group if row["conflict"] == "YES"]
        inflations = [row["time_inflation"] for row in conflicts]
        durations = [row["duration_us"] for row in conflicts]
        aggregate.append({
            "bytes_per_peer": size,
            "job": job,
            "iterations": len(group),
            "conflict_iterations": len(conflicts),
            "conflict_rate": len(conflicts) / len(group),
            "baseline_median_us": baselines[(size, job)],
            "conflict_duration_median_us": statistics.median(durations) if durations else None,
            "conflict_duration_p95_us": percentile(durations, 0.95),
            "conflict_inflation_median": statistics.median(inflations) if inflations else None,
            "conflict_inflation_p95": percentile(inflations, 0.95),
            "conflict_inflation_max": max(inflations) if inflations else None,
            "overlap_fraction_median": statistics.median(
                [row["overlap_fraction"] for row in conflicts]
            ) if conflicts else None,
            "overlapping_pairs": sum(pair_counts[(size, repeat)]
                                     for repeat in range(1, repeats + 1)),
        })

    iteration_fields = (
        "bytes_per_peer", "repeat", "job", "iteration", "started_ns", "ended_ns",
        "duration_us", "collective_window_us", "baseline_median_us", "conflict",
        "peer_iterations", "overlap_us", "overlap_fraction", "time_inflation",
        "time_inflation_percent",
    )
    aggregate_fields = (
        "bytes_per_peer", "job", "iterations", "conflict_iterations", "conflict_rate",
        "overlapping_pairs", "baseline_median_us", "conflict_duration_median_us",
        "conflict_duration_p95_us", "conflict_inflation_median",
        "conflict_inflation_p95", "conflict_inflation_max", "overlap_fraction_median",
    )
    write_csv(run_dir / "path_contention_iterations.csv", rows, iteration_fields)
    write_csv(run_dir / "path_contention_summary.csv", aggregate, aggregate_fields)
    write_csv(run_dir / "path_contention_baselines.csv", baseline_rows,
              ("bytes_per_peer", "repeat", "job", "iteration", "duration_us"))

    summary = {
        "complete": True,
        "shared_physical_directions": [f"{src}->{dst}" for src, dst in shared],
        "shared_physical_link": link_identity,
        "settings": settings,
        "aggregate": aggregate,
        "per_size_results": {
            str(size): f"results/size_{format_size_label(size)}" for size in sizes
        },
    }
    (run_dir / "path_contention_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    report = render_report(
        sizes, repeats, iterations, summary["shared_physical_directions"],
        link_identity, aggregate,
    )
    (run_dir / "path_contention_report.md").write_text(report)

    baseline_fields = ("bytes_per_peer", "repeat", "job", "iteration", "duration_us")
    for size in sizes:
        result_dir = run_dir / "results" / f"size_{format_size_label(size)}"
        size_baselines = [row for row in baseline_rows if row["bytes_per_peer"] == size]
        size_rows = [row for row in rows if row["bytes_per_peer"] == size]
        size_aggregate = [row for row in aggregate if row["bytes_per_peer"] == size]
        write_csv(result_dir / "path_contention_baselines.csv", size_baselines,
                  baseline_fields)
        write_csv(result_dir / "path_contention_iterations.csv", size_rows,
                  iteration_fields)
        write_csv(result_dir / "path_contention_summary.csv", size_aggregate,
                  aggregate_fields)
        size_summary = {
            "complete": True,
            "bytes_per_peer": size,
            "warmup_iterations": int(settings.get("warmup", 0)),
            "measured_iterations_per_case": iterations,
            "repeats": repeats,
            "profile_scope": settings.get("profile_scope", {
                "warmup_excluded": True,
                "measured_iterations_per_case": iterations,
            }),
            "shared_physical_directions": summary["shared_physical_directions"],
            "shared_physical_link": link_identity,
            "aggregate": size_aggregate,
        }
        (result_dir / "path_contention_summary.json").write_text(
            json.dumps(size_summary, indent=2, sort_keys=True) + "\n"
        )
        (result_dir / "path_contention_report.md").write_text(render_report(
            [size], repeats, iterations, summary["shared_physical_directions"],
            link_identity, size_aggregate,
        ))
    print(json.dumps({"complete": True, "aggregate_rows": len(aggregate),
                      "shared_physical_directions": summary["shared_physical_directions"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir.resolve())
    if isinstance(result, dict) and not result["complete"]:
        raise SystemExit("baseline profiles incomplete; inspect baseline_resource_cases.csv")


if __name__ == "__main__":
    main()
