#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

src_phy=""
dst_phy=""
relay_phy=""
plane=""
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
        --relay-phy) relay_phy=$2; shift 2 ;;
        --plane) plane=$2; shift 2 ;;
        --topology) topology=$2; shift 2 ;;
        --topology-json) topology_json=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --iters) iterations=$2; shift 2 ;;
        --repeats) repeats=$2; shift 2 ;;
        --timeout-seconds) timeout_seconds=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --hccn-devices) hccn_devices=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ "${src_phy}" =~ ^([0-9]|[1-5][0-9]|6[0-3])$ &&
   "${dst_phy}" =~ ^([0-9]|[1-5][0-9]|6[0-3])$ &&
   "${relay_phy}" =~ ^([0-9]|[1-5][0-9]|6[0-3])$ ]] || {
    echo "--src-phy, --dst-phy and --relay-phy must be physical device IDs 0..63" >&2
    exit 2
}
[[ "${src_phy}" != "${dst_phy}" && "${src_phy}" != "${relay_phy}" &&
   "${dst_phy}" != "${relay_phy}" ]] || {
    echo "src, dst and relay devices must be distinct" >&2
    exit 2
}
[[ "${plane}" == "0" || "${plane}" == "1" ]] || {
    echo "--plane must be exactly 0 or 1; causal comparison requires one fixed EID pair" >&2
    exit 2
}
[[ "${repeats}" =~ ^[1-9][0-9]*$ ]] || { echo "--repeats must be positive" >&2; exit 2; }
[[ -f "${topology}" ]] || { echo "topology file not found: ${topology}" >&2; exit 2; }
[[ -f "${topology_json}" ]] || {
    echo "driver topology JSON not found: ${topology_json}" >&2
    exit 2
}

run_dir="${output_root}/a5_ccu_synthetic_causal_${src_phy}_${relay_phy}_${dst_phy}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${run_dir}/cases"
manifest="${run_dir}/resolved_eid_pair.tsv"
python3 "${script_dir}/resolve_a5_synthetic_relay_eids.py" \
    --topology "${topology}" --topology-json "${topology_json}" \
    --src-phy "${src_phy}" --dst-phy "${dst_phy}" \
    --relay-phy "${relay_phy}" --plane "${plane}" >"${manifest}"

pair=$(awk -F $'\t' 'NR == 2 {print; exit}' "${manifest}")
[[ -n "${pair}" ]] || { echo "no EID pair resolved" >&2; exit 1; }
IFS=$'\t' read -r selected_plane src_die dst_die relay_die src_port relay_port_from_src \
    relay_port_to_dst dst_port src_udmac dst_udmac src_index dst_index src_eid dst_eid \
    src_edge dst_edge protocols <<<"${pair}"

printf 'case\trepeat\tbase_route\tsynthetic\tstatus\tresult\tlog\thccn_tsv\n' >"${run_dir}/case_status.tsv"
built=0

run_case() {
    local case_name=$1
    local repeat=$2
    local base_route=$3
    local synthetic=$4
    local case_dir="${run_dir}/cases/${case_name}_r${repeat}"
    local log="${case_dir}/run.log"
    local status=0 result=FAIL hccn_tsv=""
    mkdir -p "${case_dir}/hccn"
    local -a command=(bash "${script_dir}/run_a5_ccu_urma_route_probe.sh"
        --devices "${src_phy},${dst_phy}" --route-index "${base_route}"
        --bytes "${bytes}" --warmup "${warmup}" --iters "${iterations}" --remote-only
        --hccn-stat --hccn-devices "${hccn_devices}" --hccn-stat-root "${case_dir}/hccn")
    (( built == 0 )) || command+=(--skip-build)
    if (( synthetic != 0 )); then
        command+=(--rebuild-public
            --synthetic-rank0-local-eid "${src_eid}"
            --synthetic-rank0-remote-eid "${dst_eid}"
            --synthetic-rank0-local-die "${src_die}"
            --synthetic-rank0-remote-die "${dst_die}" --synthetic-hop 2)
    fi
    timeout --signal=TERM --kill-after=5 "${timeout_seconds}" \
        "${command[@]}" >"${log}" 2>&1 || status=$?
    built=1
    hccn_tsv=$(find "${case_dir}/hccn" -name hccn_counter_deltas.tsv -type f -print | head -1 || true)
    if [[ ${status} -eq 0 ]] && [[ $(grep -c 'PASS engine=CCU' "${log}" || true) -ge 2 ]]; then
        result=PASS
    elif [[ ${status} -eq 124 ]]; then
        result=TIMEOUT
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${case_name}" "${repeat}" "${base_route}" "${synthetic}" "${status}" "${result}" \
        "${log}" "${hccn_tsv}" >>"${run_dir}/case_status.tsv"
    echo "${case_name} repeat=${repeat}: ${result} status=${status} hccn=${hccn_tsv:-MISSING}"
}

for ((repeat = 1; repeat <= repeats; ++repeat)); do
    if (( repeat % 2 == 1 )); then
        run_case native0 "${repeat}" 0 0
        run_case synthetic0 "${repeat}" 0 1
        run_case native2 "${repeat}" 2 0
        run_case synthetic2 "${repeat}" 2 1
    else
        run_case synthetic2 "${repeat}" 2 1
        run_case native2 "${repeat}" 2 0
        run_case synthetic0 "${repeat}" 0 1
        run_case native0 "${repeat}" 0 0
    fi
done

python3 "${script_dir}/analyze_a5_ccu_synthetic_relay_causal_compare.py" \
    --run-dir "${run_dir}" --src-phy "${src_phy}" --dst-phy "${dst_phy}" \
    --relay-phy "${relay_phy}"

echo "Result directory: ${run_dir}"
column -s $'\t' -t "${run_dir}/case_status.tsv" 2>/dev/null || cat "${run_dir}/case_status.tsv"
