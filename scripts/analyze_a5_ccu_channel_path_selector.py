#!/usr/bin/env python3
"""Analyze ChannelDesc crossover experiments for the A5 CCU route probe."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
PREFERRED_COUNTERS = ("rx_busi_flit_num", "tx_busi_flit_num")


def parse_kv(line):
    return dict(KV_RE.findall(line))


def read_first(path, prefix):
    if not path.exists():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        if prefix in line:
            return parse_kv(line)
    return {}


def read_all(path, prefix):
    if not path.exists():
        return []
    return [parse_kv(line) for line in path.read_text(errors="replace").splitlines()
            if prefix in line]


def read_status(path, default=255):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return default


def load_meta(path):
    try:
        with path.open(newline="") as handle:
            return next(csv.DictReader(handle, delimiter="\t"))
    except (OSError, StopIteration):
        return {}


def find_hccn_table(case_dir):
    tables = sorted(case_dir.glob("hccn_routes_*/hccn_counter_deltas.tsv"))
    return tables[-1] if tables else None


def load_footprint(case_dir):
    table = find_hccn_table(case_dir)
    values = {}
    if table is None:
        return values, ""
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["counter"] not in PREFERRED_COUNTERS:
                continue
            dev = int(row["physical_device"])
            values[dev] = values.get(dev, 0.0) + abs(float(row["delta"]))
    return values, str(table)


def normalized(values, devices):
    vector = [values.get(dev, 0.0) for dev in devices]
    total = sum(vector)
    return [item / total for item in vector] if total else [0.0 for _ in vector]


def distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--candidates", default="0,2")
    parser.add_argument("--devices", default="6,7")
    args = parser.parse_args()
    candidate_a, candidate_b = [int(item) for item in args.candidates.split(",")]
    endpoint_devices = [int(item) for item in args.devices.split(",")]
    all_devices = list(range(8))

    rows = []
    for case_dir in sorted((args.run_dir / "cases").glob("*_r*")):
        meta = load_meta(case_dir / "case_meta.tsv")
        log = case_dir / "channel.log"
        variant = read_first(log, "CHANNEL_DESC_VARIANT_TRACE")
        original_desc = read_first(log, "CHANNEL_DESC_TRACE phase=before_acquire")
        catalog = read_all(log, "PATH_CATALOG")
        acquire = read_all(log, "CHANNEL_ACQUIRE_SCOPE phase=end")
        handles = read_all(log, "CHANNEL_HANDLE_TRACE phase=after_acquire")
        init_end = read_all(log, "COMM_INIT_SCOPE phase=end")
        footprint, footprint_path = load_footprint(case_dir)
        top_devices = sorted(footprint.items(), key=lambda item: item[1], reverse=True)
        rows.append({
            **meta,
            "channel_status": read_status(case_dir / "channel_status.txt"),
            "data_status": read_status(case_dir / "data_status.txt"),
            "comm_init_us": ",".join(item.get("elapsed_us", "") for item in init_end),
            "acquire_status": ",".join(item.get("status", "") for item in acquire),
            "channel_handles": ",".join(item.get("channel_handle", "") for item in handles),
            "before_raw": variant.get("before_raw", original_desc.get("channel_desc_raw", "")),
            "after_raw": variant.get("after_raw", original_desc.get("channel_desc_raw", "")),
            "local_addr": variant.get("local_addr", original_desc.get("local_addr", "")),
            "remote_addr": variant.get("remote_addr", original_desc.get("remote_addr", "")),
            "catalog_count": len(catalog),
            "footprint": footprint,
            "footprint_path": footprint_path,
            "top_devices": ",".join(f"{dev}:{value:g}" for dev, value in top_devices[:4]),
        })

    baseline_a = next((row for row in rows if row.get("case") == "c00_base_a" and
                       row["data_status"] == 0 and row["footprint"]), None)
    baseline_b = next((row for row in rows if row.get("case") == "c01_base_b" and
                       row["data_status"] == 0 and row["footprint"]), None)
    vector_a = normalized(baseline_a["footprint"], all_devices) if baseline_a else None
    vector_b = normalized(baseline_b["footprint"], all_devices) if baseline_b else None

    for row in rows:
        current = normalized(row["footprint"], all_devices)
        row["distance_to_a"] = distance(current, vector_a) if vector_a else None
        row["distance_to_b"] = distance(current, vector_b) if vector_b else None
        if not row["footprint"]:
            row["closer_footprint"] = "NO_HCCN"
        elif vector_a is None or vector_b is None:
            row["closer_footprint"] = "NO_BASELINE"
        elif abs(row["distance_to_a"] - row["distance_to_b"]) < 0.05:
            row["closer_footprint"] = "AMBIGUOUS"
        elif row["distance_to_a"] < row["distance_to_b"]:
            row["closer_footprint"] = f"candidate_{candidate_a}"
        else:
            row["closer_footprint"] = f"candidate_{candidate_b}"
        nonendpoint = sum(value for dev, value in row["footprint"].items()
                          if dev not in endpoint_devices)
        row["nonendpoint_score"] = nonendpoint

    output = args.run_dir / "parameter_causality.tsv"
    fields = ["case", "repeat", "base", "donor", "mutation", "channel_status",
              "data_status", "comm_init_us", "acquire_status", "channel_handles",
              "local_addr", "remote_addr", "distance_to_a", "distance_to_b",
              "closer_footprint", "nonendpoint_score", "top_devices", "footprint_path"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    raw_output = args.run_dir / "channel_desc_variant_raw.tsv"
    raw_fields = ["case", "repeat", "base", "donor", "mutation", "before_raw", "after_raw",
                  "local_addr", "remote_addr"]
    with raw_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "run_dir": str(args.run_dir),
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "cases": len(rows),
        "baseline_a_available": baseline_a is not None,
        "baseline_b_available": baseline_b is not None,
        "successful_channels": sum(row["channel_status"] == 0 for row in rows),
        "successful_data_cases": sum(row["data_status"] == 0 for row in rows),
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# A5 CCU Channel 路径选择因果实验报告", "",
        "## 结果概览", "",
        f"- candidate A：`{candidate_a}`", f"- candidate B：`{candidate_b}`",
        f"- case 数：{len(rows)}", f"- Channel 成功：{summary['successful_channels']}",
        f"- 数据面成功：{summary['successful_data_cases']}", "",
        "## ChannelDesc 交叉替换结果", "",
        "| case | base | donor | mutation | channel | data | footprint 更接近 | 非端点计数得分 |",
        "|---|---:|---:|---|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('case','')} r{row.get('repeat','')} | {row.get('base','')} | "
            f"{row.get('donor','')} | {row.get('mutation','')} | {row['channel_status']} | "
            f"{row['data_status']} | {row['closer_footprint']} | {row['nonendpoint_score']:g} |")
    lines += [
        "", "## 判读顺序", "",
        "1. 先看 `channel_status`：失败说明该混合 endpoint 不是合法的已注册 path key。",
        "2. Channel 成功后再看数据正确性；错误数据不能用于路径结论。",
        "3. `closer_footprint` 只表示归一化 HCCN 计数更接近哪个基线，不自动等于指定 relay。",
        "4. 如果 `c07_a_both_endpoints_from_b` 稳定复现 candidate B，endpoint pair 很可能足以选择既有 path object。",
        "5. 如果完整 endpoint pair 仍不能复现，路径还依赖 communicator side table、cache key 或私有 pathHandle。",
        "", "## 下一步内部追踪", "",
        "对第一个能稳定改变 footprint 的最小字段组合，在其 `channel.log` 对应 Acquire 时间窗内追踪：",
        "", "```text",
        "HcclChannelDesc endpoint bytes",
        "  -> first read/copy/hash",
        "  -> channel cache key",
        "  -> LinkData/HCCP private request",
        "  -> internal path object",
        "  -> handle registry",
        "  -> ChannelHandle",
        "```", "",
        "若运行时启用了 `--perf-callgraph`，各 case 的 `channel.perf.txt` 用于比较调用栈；",
        "若启用了 `--syscall-trace`，`channel.strace.*` 用于限定 Acquire 窗口中的 ioctl/connect 差异。",
        "不能把 TP handle、ChannelHandle 尾号或单次非端点计数直接命名为 route/path ID。",
    ]
    (args.run_dir / "channel_path_selector_report.md").write_text("\n".join(lines) + "\n")
    print(args.run_dir / "channel_path_selector_report.md")


if __name__ == "__main__":
    main()
