#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
devices=0,1,2,3
available=0,1,2,3,4,5,6,7
relay_map=
direct_route=0
bytes=4194304
warmup=100
iterations=20
repeats=1
timeout_seconds=600
graph=none
profile=0
cann_root=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.T560}
output_root=/home/l00934901/profiling
topology="${repo}/docs/topology/a5_hccn_device_topology_raw.txt"
topology_json=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
while (($#)); do
    case "$1" in
        --devices) devices=$2; shift 2;;
        --available-phys) available=$2; shift 2;;
        --relay-map) relay_map=$2; shift 2;;
        --direct-route) direct_route=$2; shift 2;;
        --bytes) bytes=$2; shift 2;;
        --warmup) warmup=$2; shift 2;;
        --iters) iterations=$2; shift 2;;
        --repeats) repeats=$2; shift 2;;
        --timeout-seconds) timeout_seconds=$2; shift 2;;
        --graph-backend) graph=$2; shift 2;;
        --profile) profile=1; shift;;
        --cann-root) cann_root=$2; shift 2;;
        --output-root) output_root=$2; shift 2;;
        --topology) topology=$2; shift 2;;
        --topology-json) topology_json=$2; shift 2;;
        --help) echo 'Explicit 2/4-rank AllToAll. --devices IDs [--relay-map JSON]; omit map for direct-only. --profile --graph-backend none|aclgraph'; exit 0;;
        *) echo "unsupported argument: $1" >&2; exit 2;;
    esac
done
for value in "$bytes" "$iterations" "$repeats" "$timeout_seconds"; do
    [[ "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]] || { echo 'positive numeric argument required' >&2; exit 2; }
done
[[ "$warmup" =~ ^[0-9]+$ ]] || exit 2
[[ "$graph" == none || "$graph" == aclgraph ]] || exit 2
export A5_CCU_ROUTE_PROBE_LIB=${A5_CCU_ROUTE_PROBE_LIB:-${cann_root}/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so}
export LD_LIBRARY_PATH="${cann_root}/lib64:${cann_root}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD A5_CCU_PREPARED_ALLOW_LAZY_STREAM A5_CCU_SYNTHETIC_SRC_EID A5_CCU_SYNTHETIC_DST_EID
python3 - <<'PY'
import ctypes, os
from pathlib import Path
import deep_ep.deep_ep_cpp as ext
from deep_ep import Buffer
route = ctypes.CDLL(os.environ['A5_CCU_ROUTE_PROBE_LIB'], mode=ctypes.RTLD_GLOBAL)
deep = ctypes.CDLL(str(Path(ext.__file__).resolve()))
for lib, name in ((route, 'A5CcuPeerPlanAbiVersion'), (deep, 'A5DeepEpPeerPlanAbiVersion')):
    fn = getattr(lib, name); fn.restype = ctypes.c_int
    assert fn() >= 1, f'rebuild new peer-plan ABI: {name}'
assert hasattr(Buffer, 'prepare_ccu_urma_peer_plan')
print('Verified independent peer-plan ABI 1 (route + DeepEP)', flush=True)
PY
mkdir -p "${output_root}"
run_dir=$(mktemp -d "${output_root}/a5_ccu_peer_plan_${devices//,/_}_$(date +%Y%m%d_%H%M%S).XXXXXX")
mkdir -p "${run_dir}/cases"
python3 - "$run_dir" "$repeats" <<'PY'
import json, pathlib, sys
(pathlib.Path(sys.argv[1]) / 'run_settings.json').write_text(json.dumps({'repeats': int(sys.argv[2])}) + '\n')
PY
echo "Result directory: ${run_dir}"
plan_args=(--devices "$devices" --available-phys "$available" --direct-route "$direct_route"
    --topology "$topology" --topology-json "$topology_json" --output-dir "${run_dir}/plans")
[[ -z "$relay_map" ]] || plan_args+=(--relay-map "$relay_map")
python3 "${repo}/scripts/prepare_a5_ccu_peer_plan.py" "${plan_args[@]}"
k=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["devices"]))' "${run_dir}/plans/peer_plan.json")
n=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["available_devices"]))' "${run_dir}/plans/peer_plan.json")
export ASCEND_RT_VISIBLE_DEVICES="$devices" PYTHONUNBUFFERED=1
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
profile_args=(); ((profile == 0)) || profile_args=(--profile)
failed=0
for ((r=1; r<=repeats; r++)); do
    case_dir="${run_dir}/r${r}"; mkdir -p "$case_dir"
    log="${run_dir}/cases/peer_plan_r${r}.log"
    set +e
    timeout --signal=TERM --kill-after=5 "$timeout_seconds" \
        python3 -m torch.distributed.run --standalone --nproc-per-node="$k" \
        "${repo}/tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py" \
        --manifest "${run_dir}/plans/peer_plan.tsv" --available-cards "$n" \
        --bytes "$bytes" --warmup "$warmup" --iters "$iterations" \
        --run-dir "$case_dir" --graph-backend "$graph" "${profile_args[@]}" > "$log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" > "${run_dir}/cases/peer_plan_r${r}.status"
    echo "peer_plan_r${r}: status=${rc}"
    if ((rc != 0)); then tail -n 60 "$log"; failed=1; break; fi
done
python3 "${repo}/scripts/analyze_a5_ccu_peer_plan.py" --run-dir "$run_dir"
sed -n '1,180p' "${run_dir}/peer_plan_report.md"
if ! python3 - "${run_dir}/peer_plan_summary.json" <<'PY'
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))['complete'] else 1)
PY
then
    failed=1
fi
exit "$failed"
