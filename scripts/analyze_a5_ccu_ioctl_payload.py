#!/usr/bin/env python3
"""Find ioctl payload bytes that follow the complete CommAddr pair."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PATH_CATALOG_RE = re.compile(r"PATH_CATALOG\s+(.*)")
FIELD_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


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


def endpoint_tokens(run_dir):
    result, seen = [], set()
    for log in sorted((run_dir / "cases").glob("*/run.log")):
        for line in log.read_text(errors="replace").splitlines():
            match = PATH_CATALOG_RE.search(line)
            if not match:
                continue
            fields = dict(FIELD_RE.findall(match.group(1)))
            for role in ("src", "dst"):
                raw = fields.get(f"{role}_addr", "").split("raw=", 1)[-1].replace(":", "")
                if len(raw) != 32:
                    continue
                key = (fields.get("rank", ""), fields.get("peer", ""),
                       fields.get("ordinal", ""), role, raw)
                if key in seen:
                    continue
                seen.add(key)
                result.append({"rank": key[0], "peer": key[1], "ordinal": key[2],
                               "role": role, "raw": raw, "source": str(log)})
    return result


def token_forms(raw):
    value = bytes.fromhex(raw)
    forms = [("as_printed", value), ("reverse_16", value[::-1]),
             ("reverse_each_u64", value[:8][::-1] + value[8:][::-1]),
             ("swap_u64", value[8:] + value[:8])]
    result, seen = [], set()
    for name, token in forms:
        if token not in seen:
            seen.add(token)
            result.append((name, token))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    events = load_events(args.run_dir)
    tokens = endpoint_tokens(args.run_dir)

    nested_rows = []
    for item in events:
        for nested in item.get("nested_snapshots", []):
            try:
                raw = bytes.fromhex(nested.get("hex", ""))
            except ValueError:
                raw = b""
            nested_rows.append({**item, "nested": raw,
                                "word_offset": int(nested.get("word_offset", -1)),
                                "nested_address": int(nested.get("address", 0))})

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

    window_counts = Counter()
    window_paths = defaultdict(set)
    window_callers = defaultdict(set)
    for item in events:
        label = item.get("trace_label", "")
        window = "comm_init" if label.startswith("phase=comm_init") else "channel_acquire"
        key = (window, item.get("request", ""), item.get("phase", ""))
        window_counts[key] += 1
        if item.get("fd_path"):
            window_paths[key].add(item["fd_path"])
        for frame in item.get("caller_frames", []):
            if frame.get("object"):
                window_callers[key].add(frame["object"])
    with (args.run_dir / "control_window_inventory.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["window", "request", "phase", "events", "fd_paths", "caller_objects"])
        for key in sorted(window_counts):
            writer.writerow([*key, window_counts[key], ",".join(sorted(window_paths[key])),
                             ",".join(sorted(window_callers[key]))])

    with (args.run_dir / "ioctl_nested_snapshots.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["case", "repeat", "rank", "trace_label", "request", "phase",
                         "occurrence", "word_offset", "address", "captured", "sha256", "hex"])
        for item in nested_rows:
            writer.writerow([item["case"], item["repeat"], item["rank"],
                             item.get("trace_label", ""), item.get("request", ""),
                             item.get("phase", ""), item.get("occurrence", ""),
                             item["word_offset"], hex(item["nested_address"]), len(item["nested"]),
                             hashlib.sha256(item["nested"]).hexdigest(), item["nested"].hex()])

    token_hits = []
    for item in events:
        buffers = [("top", -1, item["payload"])]
        for nested in item.get("nested_snapshots", []):
            try:
                nested_raw = bytes.fromhex(nested.get("hex", ""))
            except ValueError:
                nested_raw = b""
            buffers.append(("nested", int(nested.get("word_offset", -1)), nested_raw))
        for token in tokens:
            for representation, needle in token_forms(token["raw"]):
                for level, word_offset, haystack in buffers:
                    start = 0
                    offset = haystack.find(needle, start) if needle else -1
                    while offset >= 0:
                        token_hits.append({**token, "case": item["case"], "repeat": item["repeat"],
                                           "worker_rank": item["rank"],
                                           "trace_label": item.get("trace_label", ""),
                                           "request": item.get("request", ""),
                                           "phase": item.get("phase", ""),
                                           "fd_path": item.get("fd_path", ""),
                                           "occurrence": item.get("occurrence", ""),
                                           "level": level, "pointer_word_offset": word_offset,
                                           "byte_offset": offset, "representation": representation})
                        start = offset + 1
                        offset = haystack.find(needle, start)
    with (args.run_dir / "endpoint_token_hits.tsv").open("w", newline="") as handle:
        fields = ["case", "repeat", "worker_rank", "trace_label", "request", "phase", "fd_path",
                  "occurrence", "level", "pointer_word_offset", "byte_offset", "representation",
                  "rank", "peer", "ordinal", "role", "raw", "source"]
        writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(token_hits)

    hit_summary = Counter(
        ("comm_init" if row["trace_label"].startswith("phase=comm_init") else "channel_acquire",
         row["request"], row["fd_path"], row["ordinal"], row["role"], row["representation"],
         row["level"])
        for row in token_hits
    )
    with (args.run_dir / "endpoint_token_hit_summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["window", "request", "fd_path", "candidate_ordinal", "endpoint_role",
                         "representation", "payload_level", "hits"])
        for key, count in sorted(hit_summary.items()):
            writer.writerow([*key, count])

    with (args.run_dir / "udmac_ioctl_layout.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["case", "repeat", "worker_rank", "trace_label", "request", "phase",
                         "fd_path", "occurrence", "payload_bytes", "u64_words", "nested_count"])
        for item in events:
            if "/dev/uburma/" not in item.get("fd_path", ""):
                continue
            raw = item["payload"]
            words = [f"0x{int.from_bytes(raw[i:i + 8], 'little'):016x}"
                     for i in range(0, len(raw) - 7, 8)]
            writer.writerow([item["case"], item["repeat"], item["rank"],
                             item.get("trace_label", ""), item.get("request", ""),
                             item.get("phase", ""), item.get("fd_path", ""),
                             item.get("occurrence", ""), len(raw), ",".join(words),
                             len(item.get("nested_snapshots", []))])

    nested_groups = defaultdict(list)
    for item in nested_rows:
        nested_groups[(item.get("request", ""), item.get("phase", ""),
                       item["word_offset"], item["case"])].append({"payload": item["nested"]})
    nested_causal = []
    nested_keys = sorted({key[:3] for key in nested_groups})
    for request, phase, word_offset in nested_keys:
        modes = {case: modal_bytes(nested_groups[(request, phase, word_offset, case)])
                 for case in cases}
        width = min((len(value) for value in modes.values()), default=0)
        for offset in range(width):
            values = {case: modes[case][offset][0] for case in cases}
            confidence = min(modes[case][offset][1] for case in cases)
            if (values["candidate0"] == values["c20_addr0"] and
                    values["candidate2"] == values["c21_addr2"] and
                    values["candidate0"] != values["candidate2"]):
                nested_causal.append((request, phase, word_offset, offset,
                                      values["candidate0"], values["candidate2"], confidence))
    nested_causal.sort(key=lambda row: (-row[6], row[0], row[1], row[2], row[3]))
    with (args.run_dir / "commaddr_causal_nested_offsets.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["request", "phase", "pointer_word_offset", "nested_byte_offset",
                         "candidate0_byte", "candidate2_byte", "min_modal_confidence"])
        for row in nested_causal:
            writer.writerow([row[0], row[1], row[2], row[3], f"0x{row[4]:02x}",
                             f"0x{row[5]:02x}", f"{row[6]:.6f}"])

    high = [row for row in causal if row[5] >= 0.90]
    report = [
        "# A5 CCU ChannelAcquire 定向 ioctl payload 黑箱报告", "",
        "## 边界", "",
        "记录 communicator 初始化和单次 `HcclChannelAcquire` 两个受控时间窗；",
        "只读取 ioctl request 编码声明的顶层 `_IOC_SIZE`，以及该顶层结构中可读、对齐指针",
        "指向的有限长度快照，不递归追踪二级指针。", "",
        f"- payload 事件数：`{len(events)}`",
        f"- 满足四组 CommAddr 因果关系的字节：`{len(causal)}`",
        f"- 其中 modal confidence >= 0.90：`{len(high)}`", "",
        f"- 有界二级 pointer 快照数：`{len(nested_rows)}`",
        f"- 二级快照中的四组因果候选字节：`{len(nested_causal)}`", "",
        f"- 从 PATH_CATALOG 提取的 128-bit endpoint token：`{len(tokens)}`",
        f"- endpoint token 命中：`{len(token_hits)}`", "",
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
    comm_hits = [row for row in token_hits if row["trace_label"].startswith("phase=comm_init")]
    report += ["", "## Endpoint token 归因", ""]
    if comm_hits:
        requests = sorted({row["request"] for row in comm_hits})
        paths = sorted({row["fd_path"] for row in comm_hits})
        report += [
            f"communicator 初始化窗口命中 `{len(comm_hits)}` 次完整 endpoint token。",
            f"相关 request：`{', '.join(requests)}`；设备：`{', '.join(paths)}`。",
            "请先看 `endpoint_token_hit_summary.tsv`，再用 `endpoint_token_hits.tsv` 定位",
            "candidate ordinal、src/dst、顶层或二级 payload 以及精确字节偏移。",
            "这能把 RankGraph endpoint 收敛到具体 request，但仍不能把 token 数值直接命名为 relay ID。",
        ]
    else:
        report += [
            "communicator 初始化窗口的顶层与有限二级 payload 中均未出现完整 endpoint token。",
            "这说明 endpoint 很可能在进入当前 ioctl 前已被 HCOMM/HCCP/HAL 转换成私有对象 ID、",
            "handle 或索引。下一步应定点追踪 `/dev/uburma/*` request 的用户态构造者及其输入，",
            "而不是继续扩大 JFCE payload 或无界扫描指针。",
        ]
    report += ["", "## 输出", "", "- `ioctl_payload_inventory.tsv`：request、phase、四组事件数和设备节点。",
               "- `control_window_inventory.tsv`：comm-init与ChannelAcquire窗口的request/fd/caller。",
               "- `ioctl_payload_events.tsv`：所有有界原始快照。",
               "- `commaddr_causal_payload_offsets.tsv`：顶层payload的四组因果候选字节。",
               "- `ioctl_nested_snapshots.tsv`：顶层结构中合法指针指向的最多128字节快照。",
               "- `commaddr_causal_nested_offsets.tsv`：二级快照的四组因果候选字节。",
               "- `endpoint_token_hits.tsv`：PATH_CATALOG endpoint 在顶层/二级payload中的逐次命中。",
               "- `endpoint_token_hit_summary.tsv`：按窗口、request、设备、candidate、角色聚合的命中。",
               "- `udmac_ioctl_layout.tsv`：`/dev/uburma/*` 顶层payload的little-endian u64布局。"]
    (args.run_dir / "ioctl_payload_report.md").write_text("\n".join(report) + "\n")
    print(args.run_dir / "ioctl_payload_report.md")


if __name__ == "__main__":
    main()
