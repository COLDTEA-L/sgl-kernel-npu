#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

src_phy=""
dst_phy=""
relay_phys=""
relay_planes=""
direct_route=0
bytes=4194304
warmup=100
iterations=20
repeats=3
timeout_seconds=300
output_root=/home/l00934901/profiling
topology="${repo_root}/docs/topology/a5_hccn_device_topology_raw.txt"
topology_json=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
profile=0
profile_iters=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-phy) src_phy=$2; shift 2 ;;
        --dst-phy) dst_phy=$2; shift 2 ;;
        --relay-phys) relay_phys=$2; shift 2 ;;
        --relay-planes) relay_planes=$2; shift 2 ;;
        --direct-route) direct_route=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --iters) iterations=$2; shift 2 ;;
        --repeats) repeats=$2; shift 2 ;;
        --timeout-seconds) timeout_seconds=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --topology) topology=$2; shift 2 ;;
        --topology-json) topology_json=$2; shift 2 ;;
        --profile) profile=1; shift ;;
        --profile-iters) profile_iters=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ "${src_phy}" =~ ^[0-9]+$ && "${dst_phy}" =~ ^[0-9]+$ ]] || {
    echo "--src-phy and --dst-phy are required" >&2; exit 2;
}
IFS=',' read -ra relay_array <<<"${relay_phys}"
(( ${#relay_array[@]} == 6 )) || {
    echo "--relay-phys must explicitly list exactly six ordered relay cards" >&2; exit 2;
}
[[ "${direct_route}" =~ ^[0-9]+$ ]] || { echo "invalid --direct-route" >&2; exit 2; }
(( bytes > 0 && bytes % 256 == 0 && warmup >= 0 && iterations > 0 &&
   repeats > 0 && profile_iters > 0 )) || {
    echo "invalid bytes/warmup/iters/repeats/profile-iters" >&2; exit 2;
}

cd "${repo_root}"
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
set +u
source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash
set -u
export ASCEND_RT_VISIBLE_DEVICES="${src_phy},${dst_phy}"
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2300}
unset A5_CCU_DEBUG ASCEND_LAUNCH_BLOCKING

run_dir="${output_root}/a5_ccu_explicit_multipath_${src_phy}_${dst_phy}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${run_dir}/cases" "${run_dir}/plans" "${run_dir}/profiling"
resolved="${run_dir}/resolved_six_relays.tsv"
resolve=(python3 "${script_dir}/resolve_a5_explicit_multirelay_eids.py"
    --topology "${topology}" --topology-json "${topology_json}"
    --src-phy "${src_phy}" --dst-phy "${dst_phy}"
    --relay-phys "${relay_phys}" --weights 1,1,1,1,1,1)
[[ -z "${relay_planes}" ]] || resolve+=(--relay-planes "${relay_planes}")
"${resolve[@]}" >"${resolved}"

python3 "${script_dir}/prepare_a5_ccu_explicit_multipath_plans.py" \
    --resolved-manifest "${resolved}" --output-dir "${run_dir}/plans" \
    --direct-route "${direct_route}" --direct-relay-ratio 2:1

printf 'case\trepeat\tstatus\tresult\n' >"${run_dir}/case_status.tsv"
test_script=tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py
profile_args=()
if (( profile )); then
    profile_args=(--profile --profile-iters "${profile_iters}"
                  --profile-root "${run_dir}/profiling")
fi

run_case() {
    local name=$1 round=$2
    shift 2
    local log="${run_dir}/cases/${name}_r${round}.log" status=0 result=FAIL
    timeout --signal=TERM --kill-after=5 "${timeout_seconds}" \
      python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
        "${test_script}" --bytes "${bytes}" --warmup "${warmup}" \
        --iters "${iterations}" "${profile_args[@]}" "$@" \
        >"${log}" 2>&1 || status=$?
    grep -q '^PASS:' "${log}" && result=PASS
    printf '%s\t%s\t%s\t%s\n' "${name}" "${round}" "${status}" "${result}" \
        >>"${run_dir}/case_status.tsv"
    echo "${name}_r${round}: ${result} (status=${status})"
}

for ((round=1; round<=repeats; ++round)); do
    run_case native0 "${round}" --implementation multiroute \
        --route-index 0 --path-weights 1 --schedule concurrent
    run_case native2 "${round}" --implementation multiroute \
        --route-index 2 --path-weights 1 --schedule concurrent
    run_case direct_plus_2relay "${round}" --implementation explicit \
        --direct-route "${direct_route}" \
        --relay-manifest "${run_dir}/plans/direct_plus_2relay.tsv" \
        --path-weights 4,1,1 --schedule concurrent
    run_case direct_plus_4relay "${round}" --implementation explicit \
        --direct-route "${direct_route}" \
        --relay-manifest "${run_dir}/plans/direct_plus_4relay.tsv" \
        --path-weights 8,1,1,1,1 --schedule concurrent
    run_case direct_plus_6relay "${round}" --implementation explicit \
        --direct-route "${direct_route}" \
        --relay-manifest "${run_dir}/plans/direct_plus_6relay.tsv" \
        --path-weights 12,1,1,1,1,1,1 --schedule concurrent
done

python3 "${script_dir}/analyze_a5_ccu_explicit_multipath_perf.py" --run-dir "${run_dir}"
echo "Result directory: ${run_dir}"
column -s $'\t' -t "${run_dir}/case_status.tsv" 2>/dev/null || cat "${run_dir}/case_status.tsv"
sed -n '1,220p' "${run_dir}/explicit_multipath_perf_report.md"
