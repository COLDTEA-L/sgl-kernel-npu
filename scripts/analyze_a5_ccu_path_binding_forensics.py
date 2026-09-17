#!/usr/bin/env python3
"""Build a differential control/data-plane report for two HCCL path candidates."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


NOISE_OBJECTS = ("libascend_trace.so",)
NOISE_SYMBOLS = ("halHdcSessionConnect", "hdc_ub_connect", "drvHdcSessionConnect")
CONTROL_OBJECTS = ("libhcomm", "libhccl", "libhixl", "libascend_hal", "liburma", "liburma-udma")
LOCAL_FIELDS = {"pid", "context", "caller_frames", "trace_ts_ns", "trace_tid", "trace_label"}


def parse_kv(line):
    return dict(part.split("=", 1) for part in line.split()[1:] if "=" in part)


def load_events(root):
    events = []
    for path in root.rglob("urma_tp.pid*.jsonl"):
        for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event["_file"] = str(path)
            event["_line"] = line_no
            events.append(event)
    return events


def caller_category(event):
    frames = event.get("caller_frames") or []
    text = " ".join(
        f"{frame.get('object', '')} {frame.get('symbol', '')}" for frame in frames
    )
    if any(item in text for item in NOISE_OBJECTS + NOISE_SYMBOLS):
        return "HDC_TRACE_NOISE"
    if any(item in text for item in CONTROL_OBJECTS):
        return "COMM_CONTROL"
    return "UNKNOWN"


def normalized(event):
    value = {key: val for key, val in event.items()
             if key not in LOCAL_FIELDS and not key.startswith("_")}
    if value.get("event") == "TP_ACTIVATION":
        for key in ("target_handle", "local_jetty_handle", "active_tag",
                    "active_tx_psn", "active_rx_psn", "target_id", "remote_id"):
            value.pop(key, None)
    if value.get("event") == "GET_TP_LIST":
        value["tp_handles"] = sorted(value.get("tp_handles", []))
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def candidate_control(control_root, candidate):
    dirs = sorted(control_root.glob(f"a5_ccu_channel_trace_*/candidate_{candidate}"))
    if not dirs:
        return {"dir": "N/A", "link": {}, "desc": {}, "handle": {}, "events": []}
    directory = dirs[-1]
    result = {"dir": str(directory), "link": {}, "desc": {}, "handle": {},
              "events": load_events(directory)}
    log = directory / "channel.log"
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("COMMLINK_TRACE "):
                row = parse_kv(line)
                if row.get("rank") == "0": result["link"] = row
            elif line.startswith("CHANNEL_DESC_TRACE "):
                row = parse_kv(line)
                if row.get("rank") == "0": result["desc"] = row
            elif line.startswith("CHANNEL_HANDLE_TRACE "):
                row = parse_kv(line)
                if row.get("rank") == "0": result["handle"] = row
    return result


def event_signatures(events, name, include_noise=False):
    selected = []
    for event in events:
        if event.get("event") != name:
            continue
        if not include_noise and caller_category(event) == "HDC_TRACE_NOISE":
            continue
        selected.append(normalized(event))
    return sorted(selected)


def write_control_diff(path, candidates, controls):
    names = ["COMMLINK_ENDPOINT", "CHANNEL_DESC", "CHANNEL_HANDLE",
             "PRIVATE_TP_REQUEST", "GET_TP_LIST", "GET_TP_ATTR", "SET_TP_ATTR", "MODIFY_TP",
             "EXCHANGE_TP_INFO", "TP_ACTIVATION"]
    left, right = (controls[candidates[0]], controls[candidates[1]])
    rows = []
    for name in names:
        if name == "COMMLINK_ENDPOINT":
            a = json.dumps({key: left["link"].get(key) for key in
                            ("path_uid", "ordinal", "src_endpoint_raw", "dst_endpoint_raw",
                             "src_addr", "dst_addr")}, sort_keys=True)
            b = json.dumps({key: right["link"].get(key) for key in
                            ("path_uid", "ordinal", "src_endpoint_raw", "dst_endpoint_raw",
                             "src_addr", "dst_addr")}, sort_keys=True)
        elif name == "CHANNEL_DESC":
            a = left["desc"].get("channel_desc_raw", "N/A")
            b = right["desc"].get("channel_desc_raw", "N/A")
        elif name == "CHANNEL_HANDLE":
            a = left["handle"].get("channel_handle", "N/A")
            b = right["handle"].get("channel_handle", "N/A")
        elif name == "PRIVATE_TP_REQUEST":
            # Compare request input before the ioctl/provider call.  The after
            # image contains allocated handles and is an output observation,
            # not evidence that the selector input differed.
            a = json.dumps(sorted(
                normalized(event) for event in left["events"]
                if event.get("event") == name and event.get("phase") == "before"
                and caller_category(event) != "HDC_TRACE_NOISE"), separators=(",", ":"))
            b = json.dumps(sorted(
                normalized(event) for event in right["events"]
                if event.get("event") == name and event.get("phase") == "before"
                and caller_category(event) != "HDC_TRACE_NOISE"), separators=(",", ":"))
        else:
            a = json.dumps(event_signatures(left["events"], name), separators=(",", ":"))
            b = json.dumps(event_signatures(right["events"], name), separators=(",", ":"))
        rows.append({"layer": name, f"candidate_{candidates[0]}": a,
                     f"candidate_{candidates[1]}": b, "differs": "YES" if a != b else "NO"})
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    return rows


def digest(value):
    return hashlib.sha256((value or "").encode()).hexdigest() if value else "EMPTY"


def endpoint_tokens(control):
    tokens = []
    for key in ("src_addr", "dst_addr"):
        value = control["link"].get(key, "")
        if "raw=" in value:
            token = re.sub(r"[^0-9a-fA-F]", "", value.split("raw=", 1)[1]).lower()
            if token:
                tokens.append(token)
    return tokens


def write_private_requests(path, controls):
    rows = []
    for candidate, control in controls.items():
        tokens = endpoint_tokens(control)
        for event in control["events"]:
            if event.get("event") != "PRIVATE_TP_REQUEST":
                continue
            blobs = {
                "udata_in_hex": event.get("udata_in_hex", ""),
                "udata_out_hex": event.get("udata_out_hex", ""),
                "command_raw": event.get("command_raw", ""),
            }
            matched = []
            for token in tokens:
                for name, blob in blobs.items():
                    if token and token in blob.lower():
                        matched.append(f"{name}:{token}")
            rows.append({
                "candidate": candidate,
                "path_uid": control["link"].get("path_uid", "N/A"),
                "api": event.get("api", "N/A"),
                "phase": event.get("phase", "N/A"),
                "status": event.get("status", "N/A"),
                "local_eid": event.get("local_eid_raw", "N/A"),
                "peer_eid": event.get("peer_eid_raw", "N/A"),
                "udata_present": event.get("udata_present", False),
                "udata_in_len": event.get("udata_in_len", 0),
                "udata_out_len": event.get("udata_out_len", 0),
                "udata_in_sha256": digest(blobs["udata_in_hex"]),
                "udata_out_sha256": digest(blobs["udata_out_hex"]),
                "command_sha256": digest(blobs["command_raw"]),
                "endpoint_token_matches": ",".join(matched) or "NONE",
                "caller_category": caller_category(event),
                "trace_label": event.get("trace_label", ""),
                "source": f"{event.get('_file')}:{event.get('_line')}",
            })
    fields = ["candidate", "path_uid", "api", "phase", "status", "local_eid",
              "peer_eid", "udata_present", "udata_in_len", "udata_out_len",
              "udata_in_sha256", "udata_out_sha256", "command_sha256",
              "endpoint_token_matches", "caller_category", "trace_label", "source"]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    return rows


def write_bindings(path, controls):
    rows = []
    for candidate, control in controls.items():
        for event in control["events"]:
            if event.get("event") != "TP_ACTIVATION":
                continue
            category = caller_category(event)
            label = event.get("trace_label", "")
            confidence = "EXCLUDED_NOISE" if category == "HDC_TRACE_NOISE" else (
                "LABEL_AND_COMM_CALLER" if label and category == "COMM_CONTROL" else
                "TIME_OR_ENDPOINT_ONLY")
            rows.append({
                "candidate": candidate,
                "path_uid": control["link"].get("path_uid", "N/A"),
                "channel_handle": control["handle"].get("channel_handle", "N/A"),
                "trace_label": label,
                "event": event.get("op", "N/A"),
                "remote_eid": event.get("remote_eid_raw", event.get("target_eid_raw", "N/A")),
                "active_tp_handle": event.get("active_tp_handle", "N/A"),
                "target_tpn": event.get("target_tpn", "N/A"),
                "caller_category": category,
                "confidence": confidence,
                "source": f"{event.get('_file')}:{event.get('_line')}",
            })
    fields = ["candidate", "path_uid", "channel_handle", "trace_label", "event",
              "remote_eid", "active_tp_handle", "target_tpn", "caller_category",
              "confidence", "source"]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    return rows


def write_footprints(path, footprint_root):
    rows = []
    for delta in footprint_root.rglob("hccn_counter_deltas.tsv"):
        relative = str(delta.relative_to(footprint_root))
        case = relative.split("/", 1)[0]
        with delta.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                counter = row.get("counter", "")
                if counter in ("tx_busi_flit_num", "rx_busi_flit_num",
                               "nic_tx_all_oct_num", "nic_rx_all_oct_num"):
                    rows.append({"case": case, "physical_device": row["physical_device"],
                                 "counter": counter, "delta": row["delta"],
                                 "source": str(delta)})
    fields = ["case", "physical_device", "counter", "delta", "source"]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidates", default="0,2")
    args = parser.parse_args()
    candidates = [item.strip() for item in args.candidates.split(",")]
    if len(candidates) != 2:
        raise SystemExit("--candidates requires exactly two values")
    controls = {candidate: candidate_control(args.run_dir / "control", candidate)
                for candidate in candidates}
    diff_rows = write_control_diff(args.run_dir / "path_control_diff.tsv", candidates, controls)
    binding_rows = write_bindings(args.run_dir / "channel_resource_binding.tsv", controls)
    private_rows = write_private_requests(args.run_dir / "private_request_diff.tsv", controls)
    footprint_rows = write_footprints(args.run_dir / "physical_footprint.tsv",
                                      args.run_dir / "footprint")

    endpoint_diff = next(row["differs"] for row in diff_rows
                         if row["layer"] == "COMMLINK_ENDPOINT") == "YES"
    desc_diff = next(row["differs"] for row in diff_rows
                     if row["layer"] == "CHANNEL_DESC") == "YES"
    trusted = [row for row in binding_rows if row["confidence"] == "LABEL_AND_COMM_CALLER"]
    noise = sum(row["confidence"] == "EXCLUDED_NOISE" for row in binding_rows)
    event_diff = [row["layer"] for row in diff_rows[3:] if row["differs"] == "YES"]
    private_nonempty = [row for row in private_rows
                        if int(row["udata_in_len"] or 0) or int(row["udata_out_len"] or 0)]
    private_endpoint_match = [row for row in private_rows
                              if row["endpoint_token_matches"] != "NONE"]
    if trusted and event_diff:
        conclusion = "PROVEN_CONTROL_BINDING"
    elif endpoint_diff and desc_diff and event_diff:
        conclusion = "CONTROL_BOUNDARY_DIFFERENCE_WITHOUT_TRUSTED_TP_BINDING"
    elif endpoint_diff and desc_diff:
        conclusion = "ENDPOINT_ONLY_PRIVATE_MATCHER"
    else:
        conclusion = "NO_VISIBLE_PATH_SELECTOR"

    lines = ["# A5 CCU path binding forensic report", "",
             f"- conclusion: **{conclusion}**",
             f"- endpoint differs: **{endpoint_diff}**",
             f"- ChannelDesc differs: **{desc_diff}**",
             f"- trusted Channel/TP activation rows: **{len(trusted)}**",
             f"- excluded HDC/trace activation rows: **{noise}**",
             f"- private request rows: **{len(private_rows)}**",
             f"- non-empty provider udata rows: **{len(private_nonempty)}**",
             f"- private blobs containing CommLink endpoint token: **{len(private_endpoint_match)}**",
             f"- visible post-ChannelDesc differing boundaries: **{', '.join(event_diff) or 'none'}**",
             f"- HCCN footprint rows: **{len(footprint_rows)}**", "",
             "## Interpretation", ""]
    if conclusion == "ENDPOINT_ONLY_PRIVATE_MATCHER":
        lines.append("可见的首次差异停留在 CommLink/ChannelDesc；公开 URMA 边界未给出可信的 path→TP 分叉。路径选择应位于 HcclChannelAcquire 消费 EndpointDesc 的私有 HCCP/MUE matcher/path object 中。")
    elif conclusion == "PROVEN_CONTROL_BINDING":
        lines.append("存在带 acquire 标签且调用栈属于通信控制面的 TP activation，可继续将对应 TP/TPN 与 HCCN footprint 对齐。")
    else:
        lines.append("请结合 path_control_diff.tsv 和 channel_resource_binding.tsv 检查第一处稳定差异；禁止用 HDC/trace activation 推断 CCU Channel TP。")
    lines += ["", "## Files", "",
              "- `path_control_diff.tsv`: 控制面逐层差分。",
              "- `channel_resource_binding.tsv`: activation、调用栈分类与置信度。",
              "- `private_request_diff.tsv`: provider/CTRLQ udata、完整 ioctl command 摘要及 endpoint token 匹配。",
              "- `physical_footprint.tsv`: route0/route2/concurrent 的 HCCN 增量。",
              "- `same_process/`: 正反 acquire 顺序原始数据。"]
    (args.run_dir / "path_binding_report.md").write_text("\n".join(lines) + "\n")
    print((args.run_dir / "path_binding_report.md").resolve())


if __name__ == "__main__":
    main()
