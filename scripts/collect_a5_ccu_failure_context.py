#!/usr/bin/env python3
"""Read bounded failure context for the PIDs in one peer-plan case log."""
import argparse
from collections import deque
import os
from pathlib import Path
import re


def collect(log, roots, since=0):
    workers = set()
    markers = deque(maxlen=40)
    with Path(log).open(errors="replace") as stream:
        for line in stream:
            match = re.search(r"PEER_CASE_PHASE rank=\d+ pid=(\d+)\b", line)
            if match:
                workers.add(int(match.group(1)))
            if any(tag in line for tag in ("CCU_RUNTIME_ENV", "CCU_RUNTIME_LIBS", "CCU_PG_",
                                          "PEER_PLAN_ACQUIRE", "PEER_CASE_FAILURE")):
                markers.append(line.rstrip())
    output = [f"CCU_FAILURE_CONTEXT worker_pids={sorted(workers)}", *markers]
    seen = set()
    for root in roots:
        for pid in sorted(workers):
            for path in sorted(Path(root).rglob(f"plog-{pid}_*.log")):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    if path.stat().st_mtime < since:
                        continue
                    critical = deque(maxlen=20)
                    errors = deque(maxlen=10)
                    with path.open(errors="replace") as stream:
                        for index, line in enumerate(stream, 1):
                            entry = f"{path}:{index}: {line.rstrip()}"
                            if "HcclChannelAcquire" in line or "opExpansionMode" in line:
                                critical.append(entry)
                            elif re.search(r"ERROR|Check.*[Pp]aram|[Ii]nvalid", line):
                                errors.append(entry)
                    output.extend(critical)
                    output.extend(errors)
                except OSError as error:
                    output.append(f"CCU_PLOG_UNAVAILABLE {path}: {error}")
    if not any("plog-" in line for line in output):
        output.append("CCU_PLOG_UNAVAILABLE no recent matching worker plog; worker context retained")
    return "\n".join(output) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--since-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [Path.home() / "ascend/log"]
    if os.environ.get("ASCEND_PROCESS_LOG_PATH"):
        roots.append(Path(os.environ["ASCEND_PROCESS_LOG_PATH"]))
    result = collect(args.log, roots, args.since_file.stat().st_mtime)
    args.output.write_text(result)
    print(result, end="", flush=True)


if __name__ == "__main__":
    main()
