#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

src_phy=""
dst_phy=""
relay_phy=""
plane=all
base_route=0
topology="${repo_root}/docs/topology/a5_hccn_device_topology_raw.txt"
bytes=4096
warmup=1
iterations=3
timeout_seconds=180
output_root=/home/l00934901/profiling
channel_only=0
hccn_stat=0
hccn_devices="0,1,2,3,4,5,6,7"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-phy) src_phy=$2; shift 2 ;;
        --dst-phy) dst_phy=$2; shift 2 ;;
        --relay-phy) relay_phy=$2; shift 2 ;;
        --plane) plane=$2; shift 2 ;;
        --base-route) base_route=$2; shift 2 ;;
        --topology) topology=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --iters) iterations=$2; shift 2 ;;
        --timeout-seconds) timeout_seconds=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --channel-only) channel_only=1; shift ;;
        --hccn-stat) hccn_stat=1; shift ;;
        --hccn-devices) hccn_devices=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ "${src_phy}" =~ ^[0-7]$ && "${dst_phy}" =~ ^[0-7]$ &&
   "${relay_phy}" =~ ^[0-7]$ ]] || {
    echo "--src-phy, --dst-phy and --relay-phy must be physical device IDs 0..7" >&2
    exit 2
}
[[ "${src_phy}" != "${dst_phy}" && "${src_phy}" != "${relay_phy}" &&
   "${dst_phy}" != "${relay_phy}" ]] || {
    echo "src, dst and relay devices must be distinct" >&2
    exit 2
}
[[ -f "${topology}" ]] || { echo "topology file not found: ${topology}" >&2; exit 2; }
command -v timeout >/dev/null || { echo "GNU timeout is required" >&2; exit 2; }

run_dir="${output_root}/a5_ccu_synthetic_relay_${src_phy}_${relay_phy}_${dst_phy}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${run_dir}"
manifest="${run_dir}/resolved_eid_pairs.tsv"
python3 "${script_dir}/resolve_a5_synthetic_relay_eids.py" \
    --topology "${topology}" --src-phy "${src_phy}" --dst-phy "${dst_phy}" \
    --relay-phy "${relay_phy}" --plane "${plane}" >"${manifest}"

printf 'case\tplane\tdie\tudmac\tsrc_eid\tdst_eid\tstatus\tresult\n' >"${run_dir}/case_status.tsv"
built=0
while IFS=$'\t' read -r selected_plane route_key die udmac src_index dst_index src_eid dst_eid; do
    [[ "${selected_plane}" != "plane" ]] || continue
    case_name="plane${selected_plane}_${udmac}_src${src_index}_dst${dst_index}"
    log="${run_dir}/${case_name}.log"
    command=(bash "${script_dir}/run_a5_ccu_urma_route_probe.sh"
        --devices "${src_phy},${dst_phy}" --route-index "${base_route}"
        --rebuild-public
        --synthetic-rank0-local-eid "${src_eid}"
        --synthetic-rank0-remote-eid "${dst_eid}"
        --synthetic-die "${die}" --synthetic-hop 2
        --bytes "${bytes}" --warmup "${warmup}" --iters "${iterations}")
    (( built == 0 )) || command+=(--skip-build)
    (( channel_only == 0 )) || command+=(--channel-only)
    (( hccn_stat == 0 )) || command+=(--hccn-stat --hccn-devices "${hccn_devices}"
        --hccn-stat-root "${run_dir}/hccn")
    status=0
    timeout --signal=TERM --kill-after=5 "${timeout_seconds}" \
        "${command[@]}" >"${log}" 2>&1 || status=$?
    built=1
    if grep -q 'PASS engine=CCU' "${log}"; then
        result=PASS
    elif [[ ${status} -eq 124 ]]; then
        result=TIMEOUT
    else
        result=FAIL
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${case_name}" "${selected_plane}" "${die}" "${udmac}" "${src_eid}" "${dst_eid}" \
        "${status}" "${result}" >>"${run_dir}/case_status.tsv"
    echo "${case_name}: ${result} (status=${status})"
done <"${manifest}"

python3 "${script_dir}/analyze_a5_ccu_synthetic_relay.py" --run-dir "${run_dir}"
echo "Result directory: ${run_dir}"
column -s $'\t' -t "${run_dir}/case_status.tsv" 2>/dev/null || cat "${run_dir}/case_status.tsv"
