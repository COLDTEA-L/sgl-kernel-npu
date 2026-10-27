#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
devices=0,1,2,3
available=0,1,2,3,4,5,6,7
relay_map=
relay_phys=
relay_map_set=0
relay_phys_set=0
direct_route=0
bytes=4194304
dtype=float32
self_copy_mode=overlap
channel_sync_mode=baseline
warmup=100
iterations=20
repeats=1
timeout_seconds=600
graph=none
sync_mode=per-call
profile=0
shutdown_native=0
cann_root=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.T560}
output_root=/home/l00934901/profiling
machine_args=(--query-timeout 15)
while (($#)); do
    case "$1" in
        --devices) devices=$2; shift 2;;
        --available-phys) available=$2; shift 2;;
        --relay-map) relay_map=$2; relay_map_set=1; shift 2;;
        --relay-phys) relay_phys=$2; relay_phys_set=1; shift 2;;
        --direct-route) direct_route=$2; shift 2;;
        --bytes) bytes=$2; shift 2;;
        --dtype) dtype=$2; shift 2;;
        --self-copy-mode) self_copy_mode=$2; shift 2;;
        --channel-sync-mode) channel_sync_mode=$2; shift 2;;
        --warmup) warmup=$2; shift 2;;
        --iters) iterations=$2; shift 2;;
        --repeats) repeats=$2; shift 2;;
        --timeout-seconds) timeout_seconds=$2; shift 2;;
        --graph-backend) graph=$2; shift 2;;
        --sync-mode) sync_mode=$2; shift 2;;
        --profile) profile=1; shift;;
        --shutdown-native-backtrace) shutdown_native=1; shift;;
        --cann-root) cann_root=$2; shift 2;;
        --output-root) output_root=$2; shift 2;;
        --topology) echo "--topology TXT input removed; use --root-info, --topology-json and --hccn-tool" >&2; exit 2;;
        --root-info|--rootinfo|--topology-json|--hccn-tool|--hccl-tool|--query-timeout|--net-layer|--net-instance-id|--plane-id|--diagnostics-json) machine_args+=("$1" "$2"); shift 2;;
        --help) echo 'Explicit 2/4-rank AllToAll. --devices IDs [--relay-phys IDs (two cards) | --relay-map JSON (per-pair policy)]; omit both for direct-only. --profile --graph-backend none|aclgraph [--shutdown-native-backtrace]';
            echo '--sync-mode per-call (default) or batch (opt-in changed-payload gate; graph-backend must be none).';
            echo '--dtype float16|bfloat16|float32|int8|int32 (default float32).';
            echo '--self-copy-mode ccu (one CCU local+remote graph, grouped MS copy), ccu-direct (large-copy control), overlap (default) or serial (D2D).';
            echo '--channel-sync-mode baseline (default), packed or packed-ready; frozen at prepare.'; exit 0;;
        *) echo "unsupported argument: $1" >&2; exit 2;;
    esac
done
for value in "$bytes" "$iterations" "$repeats" "$timeout_seconds"; do
    [[ "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]] || { echo 'positive numeric argument required' >&2; exit 2; }
done
[[ "$warmup" =~ ^[0-9]+$ ]] || exit 2
case "$dtype" in float16|bfloat16|float32|int8|int32) ;; *) echo 'unsupported --dtype' >&2; exit 2;; esac
[[ "$self_copy_mode" == overlap || "$self_copy_mode" == serial || "$self_copy_mode" == ccu || "$self_copy_mode" == ccu-direct ]] || exit 2
export A5_CCU_SELF_COPY_MODE="$self_copy_mode"
case "$channel_sync_mode" in baseline|packed|packed-ready) ;; *) echo 'invalid --channel-sync-mode' >&2; exit 2;; esac
export A5_CCU_CHANNEL_SYNC_MODE="$channel_sync_mode"
[[ "$graph" == none || "$graph" == aclgraph ]] || exit 2
[[ "$sync_mode" == per-call || "$sync_mode" == batch ]] || { echo 'invalid --sync-mode' >&2; exit 2; }
[[ "$sync_mode" != batch || "$graph" == none ]] || { echo 'batch requires --graph-backend none' >&2; exit 2; }
if ((relay_map_set && relay_phys_set)); then
    echo '--relay-phys and --relay-map are mutually exclusive' >&2; exit 2
fi
# Validate the physical policy before importing any NPU runtime. Reuse the
# same argument vector for manifest generation; never hard-code a relay pool.
plan_args=(--devices "$devices" --available-phys "$available" --direct-route "$direct_route"
    "${machine_args[@]}")
((relay_map_set == 0)) || plan_args+=(--relay-map "$relay_map")
((relay_phys_set == 0)) || plan_args+=(--relay-phys "$relay_phys")
python3 "${repo}/scripts/prepare_a5_ccu_peer_plan.py" "${plan_args[@]}" --validate-only
source "${repo}/scripts/a5_ccu_test_env.sh"
a5_ccu_prepare_test_env "$cann_root" "$repo"
export A5_CCU_ROUTE_PROBE_LIB=${A5_CCU_ROUTE_PROBE_LIB:-${cann_root}/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so}
export LD_LIBRARY_PATH="${cann_root}/lib64:${cann_root}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH:-}"
unset A5_CCU_SYNTHETIC_SRC_EID A5_CCU_SYNTHETIC_DST_EID A5_CCU_SYNTHETIC_ROUTE_MANIFEST
unset A5_CCU_ROUTE_INDEX A5_CCU_ROUTE_INDICES A5_CCU_PATH_UIDS A5_CCU_PATH_WEIGHTS A5_CCU_ROUTE_SCHEDULE
timeout --signal=TERM --kill-after=5 180 \
    python3 "${repo}/tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py" --runtime-check --dtype "$dtype" --self-copy-mode "$self_copy_mode" --channel-sync-mode "$channel_sync_mode"
mkdir -p "${output_root}"
run_dir=$(mktemp -d "${output_root}/a5_ccu_peer_plan_${devices//,/_}_$(date +%Y%m%d_%H%M%S).XXXXXX")
mkdir -p "${run_dir}/cases"
python3 - "$run_dir" "$repeats" "$sync_mode" "$bytes" "$warmup" "$iterations" "$graph" "$dtype" "$self_copy_mode" "$channel_sync_mode" <<'PY'
import json, pathlib, sys
(pathlib.Path(sys.argv[1]) / 'run_settings.json').write_text(json.dumps(dict(
    repeats=int(sys.argv[2]), sync_mode=sys.argv[3], bytes_per_peer=int(sys.argv[4]),
    warmup=int(sys.argv[5]), iterations=int(sys.argv[6]), graph_backend=sys.argv[7], dtype=sys.argv[8],
    self_copy_mode=sys.argv[9], channel_sync_mode=sys.argv[10])) + '\n')
PY
echo "Result directory: ${run_dir}"
python3 "${repo}/scripts/prepare_a5_ccu_peer_plan.py" "${plan_args[@]}" --output-dir "${run_dir}/plans"
k=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["devices"]))' "${run_dir}/plans/peer_plan.json")
n=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["available_devices"]))' "${run_dir}/plans/peer_plan.json")
export ASCEND_RT_VISIBLE_DEVICES="$devices" PYTHONUNBUFFERED=1
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
profile_args=(); ((profile == 0)) || profile_args=(--profile)
failed=0
for ((r=1; r<=repeats; r++)); do
    case_dir="${run_dir}/r${r}"; mkdir -p "$case_dir"
    log="${run_dir}/cases/peer_plan_r${r}.log"
    shutdown_args=(); ((shutdown_native == 0)) || shutdown_args=(--native-backtrace)
    python3 "${repo}/scripts/watch_a5_ccu_peer_shutdown.py" \
        --case-dir "$case_dir" --log "$log" --ranks "$k" --owner-pid "$$" \
        --max-seconds "$((timeout_seconds + 20))" "${shutdown_args[@]}" \
        > "${case_dir}/shutdown_monitor.log" 2>&1 &
    monitor_pid=$!
    set +e
    timeout --signal=TERM --kill-after=5 "$timeout_seconds" \
        python3 -m torch.distributed.run --standalone --nproc-per-node="$k" \
        "${repo}/tests/python/deepep/test_a5_ccu_peer_plan_alltoall.py" \
        --manifest "${run_dir}/plans/peer_plan.tsv" --available-cards "$n" \
        --bytes "$bytes" --dtype "$dtype" --warmup "$warmup" --iters "$iterations" \
        --run-dir "$case_dir" --graph-backend "$graph" --sync-mode "$sync_mode" \
        --self-copy-mode "$self_copy_mode" --channel-sync-mode "$channel_sync_mode" "${profile_args[@]}" > "$log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" > "${run_dir}/cases/peer_plan_r${r}.status"
    # Observer exits when status appears; if collecting stacks, allow its bounded
    # capture to finish. It never signals the test or touches shared hardware.
    wait "$monitor_pid" || true
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
