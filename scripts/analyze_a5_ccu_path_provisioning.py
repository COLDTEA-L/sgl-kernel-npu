#!/usr/bin/env python3
"""Compare communicator-init and ChannelAcquire control-plane boundaries."""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
IOCTL_RE = re.compile(r"ioctl\([^,]+,\s*([^,\)]+)")


def kv(line):
    return dict(KV_RE.findall(line))


def lines_with(path, marker):
    if not path.exists():
        return []
    return [kv(line) for line in path.read_text(errors="replace").splitlines() if marker in line]


def strace_files(case_dir):
    return sorted(case_dir.glob("syscall.strace*"))


def ioctl_counts(case_dir):
    counts = Counter()
    for path in strace_files(case_dir):
        for line in path.read_text(errors="replace").splitlines():
            match = IOCTL_RE.search(line)
            if match:
                counts[match.group(1).strip()] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--candidates", default="0,1,2")
    parser.add_argument("--devices", default="2,3")
    args = parser.parse_args()

    rows = []
    request_counts = {}
    for case_dir in sorted((args.run_dir / "cases").iterdir()):
        if not case_dir.is_dir():
            continue
        log = case_dir / "run.log"
        status_path = case_dir / "status.txt"
        try:
            status = int(status_path.read_text().strip())
        except (OSError, ValueError):
            status = 255
        catalog = lines_with(log, "PATH_CATALOG")
        desc = lines_with(log, "CHANNEL_DESC_TRACE phase=before_acquire")
        variant = lines_with(log, "CHANNEL_DESC_VARIANT_TRACE")
        acquire = lines_with(log, "CHANNEL_ACQUIRE_SCOPE phase=end")
        comm = lines_with(log, "COMM_INIT_SCOPE phase=end")
        counts = ioctl_counts(case_dir)
        request_counts[case_dir.name] = counts
        chosen = variant[0] if variant else (desc[0] if desc else {})
        rows.append({
            "case": case_dir.name,
            "status": status,
            "comm_init_status": ",".join(x.get("status", "") for x in comm),
            "catalog_count": len(catalog),
            "acquire_status": ",".join(x.get("status", "") for x in acquire),
            "local_addr": chosen.get("local_addr", ""),
            "remote_addr": chosen.get("remote_addr", ""),
            "ioctl_total": sum(counts.values()),
            "ioctl_kinds": len(counts),
            "strace_files": len(strace_files(case_dir)),
        })

    with (args.run_dir / "provisioning_cases.tsv").open("w", newline="") as handle:
        fields = list(rows[0]) if rows else ["case"]
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)

    all_requests = sorted({request for counts in request_counts.values() for request in counts})
    with (args.run_dir / "ioctl_request_matrix.tsv").open("w", newline="") as handle:
        cases = sorted(request_counts)
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["request", *cases])
        for request in all_requests:
            writer.writerow([request, *(request_counts[case][request] for case in cases)])

    init_counts = request_counts.get("c00_comm_init_only", Counter())
    channel_cases = [name for name in request_counts if name.startswith("c1_candidate_")]
    differential = []
    for request in all_requests:
        channel_values = {name: request_counts[name][request] for name in channel_cases}
        if len(set(channel_values.values())) > 1 or any(
                value > init_counts[request] for value in channel_values.values()):
            differential.append((request, init_counts[request], channel_values))

    with (args.run_dir / "candidate_ioctl_differences.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["request", "comm_init_count", *channel_cases])
        for request, init_count, values in differential:
            writer.writerow([request, init_count, *(values[name] for name in channel_cases)])

    summary = {
        "run_dir": str(args.run_dir), "devices": args.devices,
        "candidates": args.candidates, "cases": len(rows),
        "differential_ioctl_requests": len(differential),
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = [
        "# A5 CCU path provisioning 取证报告", "",
        "## 实验边界", "",
        "- `c00_comm_init_only`：仅执行 communicator 初始化和销毁，不查询 RankGraph、不 Acquire Channel。",
        "- `c1_candidate_N`：新 communicator 初始化后，枚举 candidate N 并执行一次 ChannelAcquire。",
        "- `c20/c21`：固定 base ordinal，但交叉替换完整 CommAddr pair。", "",
        "## Case 概览", "",
        "| case | status | catalog | acquire | local CommAddr | remote CommAddr | ioctl total/kinds |",
        "|---|---:|---:|---|---|---|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['case']} | {row['status']} | {row['catalog_count']} | "
            f"{row['acquire_status'] or '-'} | `{row['local_addr']}` | `{row['remote_addr']}` | "
            f"{row['ioctl_total']}/{row['ioctl_kinds']} |")
    report += [
        "", "## 差分 ioctl/request 候选", "",
        f"共发现 `{len(differential)}` 个在 Channel 窗口新增或在 candidate 间计数不同的 ioctl request。",
        "详见 `candidate_ioctl_differences.tsv`；完整矩阵见 `ioctl_request_matrix.tsv`。", "",
        "## 如何使用本报告", "",
        "1. 先确认 comm-init-only 成功，且 candidate 0/2 的 ChannelAcquire 成功。",
        "2. candidate 1 若失败，将它作为‘RankGraph 可见但未 provision 可用 Channel’的负对照。",
        "3. 优先检查 candidate 0/2 计数不同、同时 c20/c21 会随 CommAddr pair 翻转的 request code。",
        "4. 用 inventory 中的 Build ID、符号和 strings 把该 request 的 caller 定位到 HCOMM/HCCP/HAL/UDMA。",
        "5. strace 只能定位 request code、fd 和调用时间，不能安全解码私有指针 payload。下一步仅对筛出的 request "
        "制作版本绑定的有界 hook，抓取入参/出参 raw bytes，再做 0/2/1 差分。", "",
        "## 可注入参数判据", "",
        "只有当某个字段同时满足以下条件，才能称为 relay provisioning 控制输入：", "",
        "- 在 communicator/path provisioning 或 Channel 创建请求中由 host 写入；",
        "- candidate 0/2 稳定不同，candidate 1 能提供负对照；",
        "- 修改后生成新的合法 CommAddr pair/path object，而非只匹配既有对象；",
        "- HCCN footprint 随指定 first-hop/relay 改变，数据正确；",
        "- 不修改全局 UBUS route table。", "",
        "在达到这些判据前，`route_addr_idx`、TPN、handle 尾号或 ioctl 偏移都只能叫候选，不能叫 path ID。",
    ]
    (args.run_dir / "path_provisioning_report.md").write_text("\n".join(report) + "\n")
    print(args.run_dir / "path_provisioning_report.md")


if __name__ == "__main__":
    main()
