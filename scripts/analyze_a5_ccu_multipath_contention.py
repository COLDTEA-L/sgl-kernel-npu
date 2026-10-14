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
from a5_ccu_path_contention_support import overlap_ns  # noqa: E402


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
    """Require P's 2--4 first edge to be Q's 4--2 first edge in reverse.

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
        raise RuntimeError(
            "P and Q resolve card 2<->4 through different physical UDie/ports; "
            "select relay planes that identify one shared physical edge"
        )
    return {
        "card2_udie": int(p_row["src_die"]),
        "card2_port": int(p_row["src_port"]),
        "card4_udie": int(p_row["relay_die"]),
        "card4_port": int(p_row["relay_port_from_src"]),
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
                directory = (run_dir / "traces" / f"size_{size}" /
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
            root = run_dir / "traces" / f"size_{size}" / f"repeat_{repeat}"
            intervals = {
                job: load_collective_intervals(root / f"concurrent_{job}", iterations)
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
            str(size): f"results/size_{size}" for size in sizes
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
        result_dir = run_dir / "results" / f"size_{size}"
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
    analyze(args.run_dir.resolve())


if __name__ == "__main__":
    main()
