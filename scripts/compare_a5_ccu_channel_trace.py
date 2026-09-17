#!/usr/bin/env python3
"""Compare CommLink, ChannelDesc and observable URMA TP traces between candidates."""

import argparse
from collections import Counter
import json
from pathlib import Path


EVENTS = (
    "GET_TP_LIST",
    "GET_TP_ATTR",
    "SET_TP_ATTR",
    "MODIFY_TP",
    "EXCHANGE_TP_INFO",
)


def fields(line):
    return dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)


def load_case(path):
    result = {"name": path.name, "commlink": [], "channel_desc": [],
              "channel_handle": [], "tp_events": []}
    log = path / "channel.log"
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("COMMLINK_TRACE "):
            result["commlink"].append(fields(line))
        elif line.startswith("CHANNEL_DESC_TRACE "):
            result["channel_desc"].append(fields(line))
        elif line.startswith("CHANNEL_HANDLE_TRACE "):
            result["channel_handle"].append(fields(line))
    for trace in path.glob("urma_tp.pid*.jsonl"):
        for line in trace.read_text(errors="replace").splitlines():
            try:
                result["tp_events"].append(json.loads(line))
            except json.JSONDecodeError:
                result["tp_events"].append({"event": "INVALID_JSON", "raw": line})
    return result


def unique_rank0(rows):
    for row in rows:
        if row.get("rank") == "0":
            return row
    return rows[0] if rows else {}


def event_counts(events):
    counts = Counter(event.get("event", "UNKNOWN") for event in events)
    return {name: counts.get(name, 0) for name in EVENTS}


def event_signature(event):
    """Remove process-local noise while retaining all path-relevant fields."""
    return json.dumps(
        {key: value for key, value in event.items() if key not in {"pid"}},
        sort_keys=True,
        separators=(",", ":"),
    )


def signatures(events, event_name):
    return sorted(
        event_signature(event) for event in events if event.get("event") == event_name
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [load_case(path) for path in sorted(args.run_dir.glob("candidate_*"))]
    if len(cases) < 2:
        raise RuntimeError("at least two candidate trace directories are required")

    lines = ["# CCU Channel 建链追踪比较", "", "## 可观测结果", "",
             "| candidate | path_uid | ordinal | endpoint pair | channel handle | GET_LIST | GET_ATTR | SET_ATTR | MODIFY | EXCHANGE |",
             "|---|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for case in cases:
        link = unique_rank0(case["commlink"])
        handle = unique_rank0(case["channel_handle"])
        pair = f"{link.get('src_addr', 'N/A')} → {link.get('dst_addr', 'N/A')}"
        counts = event_counts(case["tp_events"])
        lines.append(f"| {case['name']} | `{link.get('path_uid', 'N/A')}` | "
                     f"{link.get('ordinal', 'N/A')} | `{pair}` | "
                     f"{handle.get('channel_handle', 'N/A')} | "
                     f"{counts['GET_TP_LIST']} | {counts['GET_TP_ATTR']} | "
                     f"{counts['SET_TP_ATTR']} | {counts['MODIFY_TP']} | "
                     f"{counts['EXCHANGE_TP_INFO']} |")

    left, right = cases[:2]
    left_link, right_link = unique_rank0(left["commlink"]), unique_rank0(right["commlink"])
    left_desc, right_desc = unique_rank0(left["channel_desc"]), unique_rank0(right["channel_desc"])
    endpoint_diff = (left_link.get("src_endpoint_raw"), left_link.get("dst_endpoint_raw")) != (
        right_link.get("src_endpoint_raw"), right_link.get("dst_endpoint_raw"))
    desc_diff = left_desc.get("channel_desc_raw") != right_desc.get("channel_desc_raw")
    tp_visible = any(case["tp_events"] for case in cases)

    lines += ["", "## 首次可见差异", "",
              f"- CommLink endpoint raw 是否不同：**{'YES' if endpoint_diff else 'NO'}**",
              f"- 传入 HcclChannelAcquire 的 ChannelDesc 是否不同：**{'YES' if desc_diff else 'NO'}**",
              f"- LD_PRELOAD 是否观察到公开 URMA TP 事件：**{'YES' if tp_visible else 'NO'}**",
              ""]
    if tp_visible:
        lines.append("已观察到公开 liburma 边界事件。下面比较去除 PID 后的完整事件；handle 数值仍只用于同一次运行内关联，不能直接解释为路径编号。")
        lines += ["", "## 两个 candidate 的 URMA 边界差异", "",
                  "| event | captured | complete event signatures differ | interpretation |",
                  "|---|---:|---|---|"]
        for event_name in EVENTS:
            left_signatures = signatures(left["tp_events"], event_name)
            right_signatures = signatures(right["tp_events"], event_name)
            captured = len(left_signatures) + len(right_signatures)
            differs = left_signatures != right_signatures
            if captured == 0:
                interpretation = "该公开边界未经过或符号不可拦截"
            elif differs:
                interpretation = "该边界存在 candidate 相关差异，优先检查对应 JSONL 字段"
            else:
                interpretation = "该边界未暴露路径差异"
            lines.append(f"| {event_name} | {captured} | "
                         f"**{'YES' if differs else 'NO'}** | {interpretation} |")
    else:
        lines.append("未观察到公开 liburma 边界事件。这不表示没有 TP；表示当前 HCOMM 建链没有经过可被该 interposer 捕获的动态公开符号，可能使用内部 HCCP/MUE 接口、隐藏符号或静态绑定。")
    lines += ["", "## 边界说明", "",
              "`urma_admin list_res` 的原始 before/active/after 输出保存在各 candidate 目录。当前驱动若对 TP 和 DEV stats 均返回 `not support query`，结论只是公共资源查询面关闭；不影响已成功的 HcclChannelAcquire，也不能作为无 TP/无 relay 的证据。",
              "",
              "若 GET_TP_LIST 相同，而 SET_TP_ATTR、MODIFY_TP、EXCHANGE_TP_INFO 也均未出现 candidate 相关差异，则路径绑定发生在这些公开 liburma API 之外，下一层应追踪 HCCP/MUE 的 ChannelDesc/EndpointDesc 消费点，而不是继续尝试 `urma_admin list_res`。",
              ""]
    args.output.write_text("\n".join(lines))
    machine = {case["name"]: case for case in cases}
    (args.output.parent / "channel_trace_comparison.json").write_text(
        json.dumps(machine, indent=2, sort_keys=True))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
