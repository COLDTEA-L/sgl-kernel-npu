#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

src_phy=6
dst_phy=7
src_dev=""
dst_dev=""
src_eids="0"
dst_eids="0"
bytes=4194304
iterations=100
repeats=3
port_base=24100
control_ip=127.0.0.1
timeout_seconds=180
hccn_devices="0,1,2,3,4,5,6,7"
hccn_tool=""
perftest=""
output_root=/home/l00934901/profiling

usage() {
    cat <<'EOF'
Usage: run_a5_urma_eid_pair_scan.sh OPTIONS
  --src-phy N             Source physical NPU ID (default 6)
  --dst-phy N             Destination physical NPU ID (default 7)
  --src-dev NAME          Source URMA device, required
  --dst-dev NAME          Destination URMA device, required
  --src-eids LIST         Source per-device EID indices, e.g. 0,1,2
  --dst-eids LIST         Destination per-device EID indices
  --bytes N               Bytes per URMA Write (default 4194304)
  --iters N               Iterations per pair (default 100)
  --repeats N             Repetitions per EID pair (default 3)
  --port-base N           First TCP control port (default 24100)
  --hccn-devices LIST     Physical devices sampled by hccn_tool
  --output-root PATH      Parent output directory
  --perftest PATH         urma_perftest executable
  --hccn-tool PATH        hccn_tool executable
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-phy) src_phy=$2; shift 2 ;;
        --dst-phy) dst_phy=$2; shift 2 ;;
        --src-dev) src_dev=$2; shift 2 ;;
        --dst-dev) dst_dev=$2; shift 2 ;;
        --src-eids) src_eids=$2; shift 2 ;;
        --dst-eids) dst_eids=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --iters) iterations=$2; shift 2 ;;
        --repeats) repeats=$2; shift 2 ;;
        --port-base) port_base=$2; shift 2 ;;
        --control-ip) control_ip=$2; shift 2 ;;
        --timeout-seconds) timeout_seconds=$2; shift 2 ;;
        --hccn-devices) hccn_devices=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --perftest) perftest=$2; shift 2 ;;
        --hccn-tool) hccn_tool=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${src_dev}" && -n "${dst_dev}" ]] || {
    echo "--src-dev and --dst-dev are required" >&2
    exit 2
}
for value in "${src_phy}" "${dst_phy}" "${bytes}" "${iterations}" \
             "${repeats}" "${port_base}" "${timeout_seconds}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || { echo "numeric argument expected: ${value}" >&2; exit 2; }
done
(( src_phy != dst_phy && bytes > 0 && iterations >= 5 && repeats > 0 )) || {
    echo "invalid endpoint, byte, iteration, or repeat setting" >&2
    exit 2
}

if [[ -z "${perftest}" ]]; then
    perftest=$(command -v urma_perftest || true)
fi
[[ -n "${perftest}" && -x "${perftest}" ]] || {
    echo "urma_perftest not found; pass --perftest PATH" >&2
    exit 2
}
if [[ -z "${hccn_tool}" ]]; then
    hccn_tool=$(command -v hccn_tool || true)
    [[ -n "${hccn_tool}" ]] || hccn_tool=/usr/local/Ascend/driver/tools/hccn_tool
fi
[[ -x "${hccn_tool}" ]] || { echo "hccn_tool not executable: ${hccn_tool}" >&2; exit 2; }
command -v timeout >/dev/null || { echo "timeout command is required" >&2; exit 2; }

IFS=',' read -ra src_eid_list <<<"${src_eids}"
IFS=',' read -ra dst_eid_list <<<"${dst_eids}"
IFS=',' read -ra hccn_device_list <<<"${hccn_devices}"
for list_value in "${src_eid_list[@]}" "${dst_eid_list[@]}" "${hccn_device_list[@]}"; do
    list_value=${list_value//[[:space:]]/}
    [[ "${list_value}" =~ ^[0-9]+$ ]] || { echo "invalid comma-separated index: ${list_value}" >&2; exit 2; }
done

run_id="a5_urma_eid_scan_${src_phy}_to_${dst_phy}_$(date +%Y%m%d_%H%M%S)"
run_dir="${output_root}/${run_id}"
mkdir -p "${run_dir}"

{
    echo "run_id=${run_id}"
    echo "src_phy=${src_phy}"
    echo "dst_phy=${dst_phy}"
    echo "src_dev=${src_dev}"
    echo "dst_dev=${dst_dev}"
    echo "src_eids=${src_eids}"
    echo "dst_eids=${dst_eids}"
    echo "bytes=${bytes}"
    echo "iterations=${iterations}"
    echo "repeats=${repeats}"
    echo "hccn_devices=${hccn_devices}"
    echo "perftest=${perftest}"
    echo "hccn_tool=${hccn_tool}"
} >"${run_dir}/run_config.env"

if command -v urma_admin >/dev/null 2>&1; then
    urma_admin show >"${run_dir}/urma_admin_show.txt" 2>&1 || true
fi
printf 'pair_id\tsrc_phy\tdst_phy\tsrc_dev\tdst_dev\tsrc_eid_idx\tdst_eid_idx\trepeat\tport\tstatus\tpair_dir\n' \
    >"${run_dir}/pairs.tsv"

capture_hccn() {
    local phase=$1
    local pair_dir=$2
    local device
    for device in "${hccn_device_list[@]}"; do
        device=${device//[[:space:]]/}
        {
            echo "# command: ${hccn_tool} -i ${device} -stat -g"
            echo "# timestamp: $(date --iso-8601=ns)"
            env -u ASCEND_RT_VISIBLE_DEVICES -u ASCEND_VISIBLE_DEVICES \
                "${hccn_tool}" -i "${device}" -stat -g
        } >"${pair_dir}/${phase}_device${device}.txt" 2>&1 || true
    done
}

pair_number=0
for src_eid in "${src_eid_list[@]}"; do
    src_eid=${src_eid//[[:space:]]/}
    for dst_eid in "${dst_eid_list[@]}"; do
        dst_eid=${dst_eid//[[:space:]]/}
        for ((repeat = 1; repeat <= repeats; ++repeat)); do
            pair_id="src${src_eid}_dst${dst_eid}_r${repeat}"
            pair_dir="${run_dir}/${pair_id}"
            mkdir -p "${pair_dir}"
            port=$((port_base + pair_number))
            pair_number=$((pair_number + 1))
            status=PASS

            echo "===== ${pair_id}: ${src_dev}/eid${src_eid} -> ${dst_dev}/eid${dst_eid} ====="
            capture_hccn before "${pair_dir}"

            server_cmd=("${perftest}" write_bw -d "${dst_dev}" --eid_idx "${dst_eid}"
                --ctp -s "${bytes}" -n "${iterations}" -w -P "${port}")
            client_cmd=("${perftest}" write_bw -d "${src_dev}" --eid_idx "${src_eid}"
                --ctp -s "${bytes}" -n "${iterations}" -w -P "${port}" -S "${control_ip}")
            printf '%q ' "${server_cmd[@]}" >"${pair_dir}/server_command.txt"
            printf '\n' >>"${pair_dir}/server_command.txt"
            printf '%q ' "${client_cmd[@]}" >"${pair_dir}/client_command.txt"
            printf '\n' >>"${pair_dir}/client_command.txt"

            timeout "${timeout_seconds}s" "${server_cmd[@]}" \
                >"${pair_dir}/server.log" 2>&1 &
            server_pid=$!
            sleep 1
            client_rc=0
            timeout "${timeout_seconds}s" "${client_cmd[@]}" \
                >"${pair_dir}/client.log" 2>&1 || client_rc=$?
            if (( client_rc != 0 )); then
                kill "${server_pid}" 2>/dev/null || true
            fi
            server_rc=0
            wait "${server_pid}" || server_rc=$?
            if (( client_rc != 0 || server_rc != 0 )); then
                status=FAIL
                echo "pair failed: client_rc=${client_rc} server_rc=${server_rc}" >&2
            fi
            capture_hccn after "${pair_dir}"

            python3 "${repo_root}/scripts/analyze_a5_urma_eid_pairs.py" \
                --snapshot-dir "${pair_dir}" \
                --snapshot-only \
                --output "${pair_dir}/hccn_counter_deltas.tsv" || status=FAIL
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${pair_id}" "${src_phy}" "${dst_phy}" "${src_dev}" "${dst_dev}" \
                "${src_eid}" "${dst_eid}" "${repeat}" "${port}" "${status}" "${pair_dir}" \
                >>"${run_dir}/pairs.tsv"
        done
    done
done

python3 "${repo_root}/scripts/analyze_a5_urma_eid_pairs.py" \
    --scan-dir "${run_dir}" \
    --endpoint-devices "${src_phy},${dst_phy}" \
    --output "${run_dir}/eid_pair_relay_map.tsv"

echo "EID-pair scan completed: ${run_dir}"
echo "Relay map: ${run_dir}/eid_pair_relay_map.tsv"
