#!/usr/bin/env python3
"""Find ioctl payload bytes that follow the complete CommAddr pair."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_events(run_dir):
    events = []
    for case_dir in sorted((run_dir / "cases").iterdir()):
        if not case_dir.is_dir():
            continue
        stem = case_dir.name.rsplit("_r", 1)
        case = stem[0]
        repeat = int(stem[1]) if len(stem) == 2 and stem[1].isdigit() else 0
        for path in sorted(case_dir.glob("ioctl_payload.rank*.pid*.jsonl")):
            rank = path.name.split(".rank", 1)[1].split(".", 1)[0]
            for line in path.read_text(errors="replace").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") != "IOCTL_PAYLOAD":
                    continue
                raw = event.get("payload_hex", "")
                try:
                    payload = bytes.fromhex(raw)
                except ValueError:
                    payload = b""
                event.update(case=case, repeat=repeat, rank=rank,
                             source=str(path), payload=payload)
                events.append(event)
    return events


def modal_bytes(records):
    if not records:
        return []
    width = max(len(item["payload"]) for item in records)
    result = []
    for offset in range(width):
        values = [item["payload"][offset] for item in records if len(item["payload"]) > offset]
        if not values:
            result.append((None, 0.0))
            continue
        value, count = Counter(values).most_common(1)[0]
        result.append((value, count / len(values)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    events = load_events(args.run_dir)

    with (args.run_dir / "ioctl_payload_events.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["case", "repeat", "rank", "request", "phase", "occurrence",
                         "fd_path", "ioc_size", "captured", "sha256", "payload_hex",
                         "caller_frames_json", "trace_label", "source"])
        for item in events:
            writer.writerow([item["case"], item["repeat"], item["rank"], item.get("request", ""),
                             item.get("phase", ""), item.get("occurrence", ""),
                             item.get("fd_path", ""), item.get("ioc_size", ""),
                             len(item["payload"]), hashlib.sha256(item["payload"]).hexdigest(),
                             item["payload"].hex(), json.dumps(item.get("caller_frames", [])),
                             item.get("trace_label", ""), item["source"]])

    groups = defaultdict(list)
    for item in events:
        groups[(item.get("request", ""), item.get("phase", ""), item["case"])].append(item)

    cases = ("candidate0", "candidate2", "c20_addr0", "c21_addr2")
    causal = []
    keys = sorted({(item.get("request", ""), item.get("phase", "")) for item in events})
    for request, phase in keys:
        modes = {case: modal_bytes(groups[(request, phase, case)]) for case in cases}
        width = min((len(value) for value in modes.values()), default=0)
        for offset in range(width):
            values = {case: modes[case][offset][0] for case in cases}
            confidence = min(modes[case][offset][1] for case in cases)
            follows_pair = (values["candidate0"] == values["c20_addr0"] and
                            values["candidate2"] == values["c21_addr2"] and
                            values["candidate0"] != values["candidate2"])
            if follows_pair:
                causal.append((request, phase, offset, values["candidate0"],
                               values["candidate2"], confidence,
                               *(len(groups[(request, phase, case)]) for case in cases)))

    causal.sort(key=lambda row: (-row[5], row[0], row[1], row[2]))
    with (args.run_dir / "commaddr_causal_payload_offsets.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["request", "phase", "byte_offset", "candidate0_byte", "candidate2_byte",
                         "min_modal_confidence", *[f"{case}_samples" for case in cases]])
        for row in causal:
            writer.writerow([row[0], row[1], row[2], f"0x{row[3]:02x}", f"0x{row[4]:02x}",
                             f"{row[5]:.6f}", *row[6:]])

    inventory = []
    for request, phase in keys:
        counts = [len(groups[(request, phase, case)]) for case in cases]
        paths = sorted({item.get("fd_path", "") for case in cases
                        for item in groups[(request, phase, case)] if item.get("fd_path")})
        inventory.append((request, phase, *counts, ",".join(paths)))
    with (args.run_dir / "ioctl_payload_inventory.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["request", "phase", *[f"{case}_events" for case in cases], "fd_paths"])
        writer.writerows(inventory)

    high = [row for row in causal if row[5] >= 0.90]
    report = [
        "# A5 CCU ChannelAcquire 定向 ioctl payload 黑箱报告", "",
        "## 边界", "",
        "仅记录 `A5UrmaTpTraceSetLabel()` 标记的单次 `HcclChannelAcquire` 时间窗；",
        "只读取 ioctl request 编码声明的顶层 `_IOC_SIZE`，不递归解引用未知指针。", "",
        f"- payload 事件数：`{len(events)}`",
        f"- 满足四组 CommAddr 因果关系的字节：`{len(causal)}`",
        f"- 其中 modal confidence >= 0.90：`{len(high)}`", "",
        "四组关系为：`candidate0 == c20_addr0`、`candidate2 == c21_addr2`，且两组不同。", "",
        "## 判读", "",
    ]
    if high:
        report += [
            "存在高稳定候选偏移。先在 `commaddr_causal_payload_offsets.tsv` 中按 request/phase 聚类，",
            "再结合 `ioctl_payload_events.tsv` 的 fd、调用栈和原始 payload 确认结构边界。",
            "这些偏移仍只是 selector/path-object 候选，修改前必须确定 ABI 与校验字段。",
        ]
    else:
        report += [
            "顶层 ioctl payload 未暴露稳定的 CommAddr/path 差异。下一层应根据本次记录的 caller/fd，",
            "对已确认的顶层结构中指针字段做 Build-ID 绑定的定点解码，或转向 ioctl 之前的",
            "HCCP/MUE request builder；不能扩大为任意指针扫描。",
        ]
    report += ["", "## 输出", "", "- `ioctl_payload_inventory.tsv`：request、phase、四组事件数和设备节点。",
               "- `ioctl_payload_events.tsv`：所有有界原始快照。",
               "- `commaddr_causal_payload_offsets.tsv`：四组因果差分后的候选字节。"]
    (args.run_dir / "ioctl_payload_report.md").write_text("\n".join(report) + "\n")
    print(args.run_dir / "ioctl_payload_report.md")


if __name__ == "__main__":
    main()
