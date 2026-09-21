#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

src_phy=""
dst_phy=""
relay_phys=""
relay_planes=""
weights=""
base_route=0
topology="${repo_root}/docs/topology/a5_hccn_device_topology_raw.txt"
topology_json=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
bytes=4194304
warmup=10
iterations=100
repeats=3
timeout_seconds=300
output_root=/home/l00934901/profiling
hccn_devices="0,1,2,3,4,5,6,7"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-phy) src_phy=$2; shift 2 ;;
        --dst-phy) dst_phy=$2; shift 2 ;;
        --relay-phys) relay_phys=$2; shift 2 ;;
        --relay-planes) relay_planes=$2; shift 2 ;;
        --weights) weights=$2; shift 2 ;;
        --base-route) base_route=$2; shift 2 ;;
        --topology) topology=$2; shift 2 ;;
        --topology-json) topology_json=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --iters) iterations=$2; shift 2 ;;
        --repeats) repeats=$2; shift 2 ;;
        --timeout-seconds) timeout_seconds=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --hccn-devices) hccn_devices=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ "${src_phy}" =~ ^[0-9]+$ && "${dst_phy}" =~ ^[0-9]+$ ]] || {
    echo "--src-phy and --dst-phy are required physical device IDs" >&2; exit 2;
}
IFS=',' read -ra relay_array <<<"${relay_phys}"
(( ${#relay_array[@]} >= 2 )) || {
    echo "--relay-phys must explicitly name at least two relay cards, for example 4,5" >&2; exit 2;
}
for relay in "${relay_array[@]}"; do
    [[ "${relay}" =~ ^[0-9]+$ ]] || { echo "invalid relay card: ${relay}" >&2; exit 2; }
done
if [[ -z "${weights}" ]]; then
    weights=$(printf '1,%.0s' "${relay_array[@]}")
    weights=${weights%,}
fi
IFS=',' read -ra weight_array <<<"${weights}"
(( ${#weight_array[@]} == ${#relay_array[@]} )) || {
    echo "--weights must contain one value per explicit relay" >&2; exit 2;
}
for weight in "${weight_array[@]}"; do
    [[ "${weight}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid weight: ${weight}" >&2; exit 2; }
done
(( bytes > 0 && bytes % 256 == 0 && iterations > 0 && repeats > 0 )) || {
    echo "bytes must be a positive multiple of 256; iters/repeats must be positive" >&2; exit 2;
}
[[ -f "${topology}" && -f "${topology_json}" ]] || {
    echo "topology inventory or driver JSON is missing" >&2; exit 2;
}

cd "${repo_root}"
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
# The installed DeepEP vendor script prepends to ASCEND_CUSTOM_OPP_PATH without
# a ${var:-} guard.  It is safe to source, but is not nounset-clean.
set +u
source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash
set -u
export ASCEND_RT_VISIBLE_DEVICES="${src_phy},${dst_phy}"
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2300}
export PYTHONUNBUFFERED=1
unset ASCEND_LAUNCH_BLOCKING A5_CCU_SOURCE_ROUTE_MANIFEST A5_CCU_SOURCE_ROUTE_PROVIDER
unset A5_CCU_SYNTHETIC_RANK0_LOCAL_EID A5_CCU_SYNTHETIC_RANK0_REMOTE_EID
unset A5_CCU_PATH_UIDS A5_CCU_ROUTE_INDICES A5_CCU_SYNTHETIC_ROUTE_MANIFEST

run_dir="${output_root}/a5_ccu_explicit_multirelay_${src_phy}_${dst_phy}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${run_dir}/cases" "${run_dir}/footprint"
raw_manifest="${run_dir}/resolved_explicit_relays.tsv"
resolve=(python3 "${script_dir}/resolve_a5_explicit_multirelay_eids.py"
    --topology "${topology}" --topology-json "${topology_json}"
    --src-phy "${src_phy}" --dst-phy "${dst_phy}"
    --relay-phys "${relay_phys}" --weights "${weights}")
[[ -z "${relay_planes}" ]] || resolve+=(--relay-planes "${relay_planes}")
"${resolve[@]}" >"${raw_manifest}"
python3 "${script_dir}/analyze_a5_ccu_explicit_multirelay.py" \
    --prepare --run-dir "${run_dir}" --manifest "${raw_manifest}" >/dev/null

reversed_weights=""
for ((i=${#weight_array[@]}-1; i>=0; --i)); do
    reversed_weights+="${weight_array[i]},"
done
reversed_weights=${reversed_weights%,}
test_script=tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py
printf 'case\tstatus\tresult\n' >"${run_dir}/case_status.tsv"

run_python_case() {
    local name=$1 manifest=$2 schedule=$3 case_weights=$4
    local log="${run_dir}/cases/${name}.log" status=0 result=FAIL
    timeout --signal=TERM --kill-after=5 "${timeout_seconds}" \
      python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
        "${test_script}" --implementation multiroute --route-index "${base_route}" \
        --synthetic-route-manifest "${manifest}" --path-weights "${case_weights}" \
        --schedule "${schedule}" --bytes "${bytes}" --warmup "${warmup}" \
        --iters "${iterations}" >"${log}" 2>&1 || status=$?
    grep -q '^PASS: implementation=multiroute' "${log}" && result=PASS
    printf '%s\t%s\t%s\n' "${name}" "${status}" "${result}" >>"${run_dir}/case_status.tsv"
    echo "${name}: ${result} (status=${status})"
}

for ((round=1; round<=repeats; ++round)); do
    for relay in "${relay_array[@]}"; do
        run_python_case "single_relay${relay}_r${round}" \
            "${run_dir}/manifests/relay_${relay}.tsv" concurrent 1
    done
    if (( round % 2 == 1 )); then
        run_python_case "serial_r${round}" "${run_dir}/manifests/all.tsv" serial "${weights}"
        run_python_case "concurrent_r${round}" "${run_dir}/manifests/all.tsv" concurrent "${weights}"
    else
        run_python_case "concurrent_r${round}" "${run_dir}/manifests/all.tsv" concurrent "${weights}"
        run_python_case "serial_r${round}" "${run_dir}/manifests/all.tsv" serial "${weights}"
    fi
    run_python_case "concurrent_reverse_r${round}" \
        "${run_dir}/manifests/all_reversed.tsv" concurrent "${reversed_weights}"
done

footprint_log="${run_dir}/footprint/run.log"
footprint_status=0
timeout --signal=TERM --kill-after=5 "${timeout_seconds}" \
  bash "${script_dir}/run_a5_ccu_urma_route_probe.sh" \
    --devices "${src_phy},${dst_phy}" --route-index "${base_route}" \
    --synthetic-route-manifest "${run_dir}/manifests/all.tsv" \
    --path-weights "${weights}" --bytes "${bytes}" --warmup "${warmup}" \
    --iters "${iterations}" --remote-only --hccn-stat \
    --hccn-devices "${hccn_devices}" --hccn-stat-root "${run_dir}/footprint" \
    >"${footprint_log}" 2>&1 || footprint_status=$?
footprint_result=FAIL
grep -q 'PASS engine=CCU' "${footprint_log}" && footprint_result=PASS
printf 'footprint\t%s\t%s\n' "${footprint_status}" "${footprint_result}" \
    >>"${run_dir}/case_status.tsv"
echo "footprint: ${footprint_result} (status=${footprint_status})"

python3 "${script_dir}/analyze_a5_ccu_explicit_multirelay.py" --run-dir "${run_dir}"
echo "Result directory: ${run_dir}"
column -s $'\t' -t "${run_dir}/case_status.tsv" 2>/dev/null || cat "${run_dir}/case_status.tsv"
sed -n '1,260p' "${run_dir}/explicit_multirelay_report.md"
