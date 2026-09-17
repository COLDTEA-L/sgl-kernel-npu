#!/usr/bin/env bash
set -euo pipefail

devices="4,5"
bytes=2097152
warmup=100
iters=100
path_uids=""
path_weights="2,1"
output_root=/home/l00934901/profiling

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) devices=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --iters) iters=$2; shift 2 ;;
        --path-uids) path_uids=$2; shift 2 ;;
        --path-weights) path_weights=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash
export ASCEND_RT_VISIBLE_DEVICES="$devices"
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2300}
export PYTHONUNBUFFERED=1
unset ASCEND_LAUNCH_BLOCKING A5_CCU_SOURCE_ROUTE_MANIFEST A5_CCU_SOURCE_ROUTE_PROVIDER

run_dir="${output_root}/a5_ccu_discovered_path_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
test_script=tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py

run_case() {
    local name=$1
    shift
    echo "===== ${name} ====="
    python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
        "$test_script" "$@" --bytes "$bytes" --warmup "$warmup" --iters "$iters" \
        2>&1 | tee "$run_dir/${name}.log"
}

if [[ -z "$path_uids" ]]; then
    echo "--path-uids is required. First generate PATH_CATALOG as documented." >&2
    exit 2
fi
IFS=',' read -r path_a path_b extra <<<"$path_uids"
if [[ -z "${path_a:-}" || -z "${path_b:-}" || -n "${extra:-}" ]]; then
    echo "this two-path experiment requires exactly two --path-uids" >&2
    exit 2
fi
IFS=',' read -r weight_a weight_b extra_weight <<<"$path_weights"
if [[ -z "${weight_a:-}" || -z "${weight_b:-}" || -n "${extra_weight:-}" ]]; then
    echo "--path-weights must contain exactly two integers" >&2
    exit 2
fi

run_case native --implementation native
run_case path_a --implementation multiroute --path-uids "$path_a" --path-weights 1
run_case path_b --implementation multiroute --path-uids "$path_b" --path-weights 1
run_case serial --implementation multiroute --path-uids "$path_uids" \
    --path-weights "$path_weights" --schedule serial
run_case concurrent --implementation multiroute --path-uids "$path_uids" \
    --path-weights "$path_weights" --schedule concurrent

python3 - "$run_dir" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.log")):
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("RESULT_JSON "):
            row = json.loads(line[len("RESULT_JSON "):])
            row["case"] = path.stem
            rows.append(row)
            break
if len(rows) != 5:
    raise SystemExit(f"expected five successful cases, found {len(rows)}")
serial = next(row for row in rows if row["case"] == "serial")["host_batch_avg_us"]
concurrent = next(row for row in rows if row["case"] == "concurrent")["host_batch_avg_us"]
best_single = min(row["host_batch_avg_us"] for row in rows if row["case"] in ("path_a", "path_b"))
summary = {"cases": rows, "concurrent_faster_than_serial": concurrent < serial,
           "concurrent_faster_than_best_single": concurrent < best_single,
           "output_dir": str(root.resolve())}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print("DISCOVERED_PATH_SUMMARY " + json.dumps(summary, sort_keys=True))
PY

echo "Results: ${run_dir}"
