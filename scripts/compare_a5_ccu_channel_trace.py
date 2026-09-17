#!/usr/bin/env python3
"""Compare CommLink, ChannelDesc and observable URMA TP traces between candidates."""

import argparse
import json
import re
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [load_case(path) for path in sorted(args.run_dir.glob("candidate_*"))]
    if len(cases) < 2:
        raise RuntimeError("at least two candidate trace directories are required")

    lines = ["# CCU Channel 建链追踪比较", "", "## 可观测结果", "",
             "| candidate | path_uid | ordinal | endpoint pair | channel handle | GET_TP_LIST events |",
             "|---|---|---:|---|---|---:|"]
    for case in cases:
        link = unique_rank0(case["commlink"])
        handle = unique_rank0(case["channel_handle"])
        pair = f"{link.get('src_addr', 'N/A')} → {link.get('dst_addr', 'N/A')}"
        lines.append(f"| {case['name']} | `{link.get('path_uid', 'N/A')}` | "
                     f"{link.get('ordinal', 'N/A')} | `{pair}` | "
                     f"{handle.get('channel_handle', 'N/A')} | {len(case['tp_events'])} |")

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
              f"- LD_PRELOAD 是否观察到 GET_TP_LIST/GET_TP_ATTR：**{'YES' if tp_visible else 'NO'}**",
              ""]
    if tp_visible:
        lines.append("已观察到公开 liburma 边界事件；下一步按 local/peer EID 和 TP handle 对齐两个 candidate。")
    else:
        lines.append("未观察到公开 liburma 边界事件。这不表示没有 TP；表示当前 HCOMM 建链没有经过可被该 interposer 捕获的动态公开符号，可能使用内部 HCCP/MUE 接口、隐藏符号或静态绑定。")
    lines += ["", "## 边界说明", "",
              "`urma_admin list_res` 的原始 before/active/after 输出保存在各 candidate 目录。若驱动返回 not support，TPN/TPG、route_addr_idx 和内部 path object 仍为不可见，不能从 handle 数值反推。",
              ""]
    args.output.write_text("\n".join(lines))
    machine = {case["name"]: case for case in cases}
    (args.output.parent / "channel_trace_comparison.json").write_text(
        json.dumps(machine, indent=2, sort_keys=True))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
