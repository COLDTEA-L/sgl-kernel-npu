#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
test_dir="${repo_root}/examples/a5_ccu_urma_route_probe/testcase"

devices="2,3"
route_index=0
route_indices=""
source_route_manifest=""
source_route_provider=""
bytes=2097152
warmup=10
iterations=100
profile=0
remote_only=0
channel_only=0
comm_init_only=0
skip_build=0
sweep=0
profile_root=/home/l00934901/profiling
hccn_stat=0
hccn_devices="0,1,2,3,4,5,6,7"
hccn_tool_path=""
hccn_stat_root=""
descriptor_mutation=""
descriptor_donor=""
worker_preload=""
worker_trace_prefix=""
synthetic_rank0_local_eid=""
synthetic_rank0_remote_eid=""
synthetic_die=""
synthetic_hop=2
rebuild_public=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) devices=$2; shift 2 ;;
        --route-index) route_index=$2; shift 2 ;;
        --route-indices) route_indices=$2; shift 2 ;;
        --source-route-manifest) source_route_manifest=$2; shift 2 ;;
        --source-route-provider) source_route_provider=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --iters) iterations=$2; shift 2 ;;
        --remote-only) remote_only=1; shift ;;
        --channel-only) channel_only=1; shift ;;
        --comm-init-only) comm_init_only=1; shift ;;
        --skip-build) skip_build=1; shift ;;
        --sweep) sweep=1; shift ;;
        --profile) profile=1; shift ;;
        --profile-root) profile_root=$2; shift 2 ;;
        --hccn-stat) hccn_stat=1; shift ;;
        --hccn-devices) hccn_devices=$2; shift 2 ;;
        --hccn-tool) hccn_tool_path=$2; shift 2 ;;
        --hccn-stat-root) hccn_stat_root=$2; shift 2 ;;
        --descriptor-mutation) descriptor_mutation=$2; shift 2 ;;
        --descriptor-donor) descriptor_donor=$2; shift 2 ;;
        --worker-preload) worker_preload=$2; shift 2 ;;
        --worker-trace-prefix) worker_trace_prefix=$2; shift 2 ;;
        --synthetic-rank0-local-eid) synthetic_rank0_local_eid=$2; shift 2 ;;
        --synthetic-rank0-remote-eid) synthetic_rank0_remote_eid=$2; shift 2 ;;
        --synthetic-die) synthetic_die=$2; shift 2 ;;
        --synthetic-hop) synthetic_hop=$2; shift 2 ;;
        --rebuild-public) rebuild_public=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

(( remote_only == 0 || channel_only == 0 )) || {
    echo "--remote-only and --channel-only cannot be used together" >&2
    exit 2
}
(( comm_init_only == 0 || (remote_only == 0 && channel_only == 0) )) || {
    echo "--comm-init-only cannot be combined with --remote-only or --channel-only" >&2
    exit 2
}
(( sweep == 0 || channel_only == 0 )) || {
    echo "--sweep and --channel-only cannot be used together" >&2
    exit 2
}
(( sweep == 0 || profile == 0 )) || {
    echo "--sweep and --profile cannot be used together; profile one size at a time" >&2
    exit 2
}

if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# This is an HCCL custom package installed under the active CANN tree.  A
# DeepEP vendor environment may redirect ASCEND_CUSTOM_OPP_PATH elsewhere.
unset ASCEND_CUSTOM_OPP_PATH

[[ "${devices}" == *,* && "${devices}" != *,*,* ]] || {
    echo "--devices must contain exactly two physical device IDs, for example 2,3" >&2
    exit 2
}

export ASCEND_RT_VISIBLE_DEVICES="${devices}"
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export A5_CCU_ROUTE_INDEX="${route_index}"
if [[ -n "${route_indices}" ]]; then
    export A5_CCU_ROUTE_INDICES="${route_indices}"
else
    unset A5_CCU_ROUTE_INDICES
fi
if [[ -n "${source_route_manifest}" ]]; then
    export A5_CCU_SOURCE_ROUTE_MANIFEST="${source_route_manifest}"
    export A5_CCU_SOURCE_ROUTE_PROVIDER="${source_route_provider}"
else
    unset A5_CCU_SOURCE_ROUTE_MANIFEST A5_CCU_SOURCE_ROUTE_PROVIDER
fi
if [[ -n "${descriptor_mutation}" ]]; then
    [[ -n "${descriptor_donor}" ]] || {
        echo "--descriptor-mutation requires --descriptor-donor" >&2
        exit 2
    }
    [[ -z "${route_indices}" ]] || {
        echo "descriptor mutation only supports --route-index" >&2
        exit 2
    }
    export A5_CCU_DESC_MUTATION="${descriptor_mutation}"
    export A5_CCU_DESC_DONOR_ROUTE="${descriptor_donor}"
else
    unset A5_CCU_DESC_MUTATION A5_CCU_DESC_DONOR_ROUTE
fi
if [[ -n "${synthetic_rank0_local_eid}" || -n "${synthetic_rank0_remote_eid}" ]]; then
    [[ -n "${synthetic_rank0_local_eid}" && -n "${synthetic_rank0_remote_eid}" &&
       "${synthetic_die}" =~ ^[01]$ ]] || {
        echo "synthetic mode requires both rank0 EIDs and --synthetic-die 0|1" >&2
        exit 2
    }
    [[ -z "${route_indices}" ]] || {
        echo "synthetic mode currently supports one --route-index only" >&2
        exit 2
    }
    export A5_CCU_SYNTHETIC_RANK0_LOCAL_EID="${synthetic_rank0_local_eid}"
    export A5_CCU_SYNTHETIC_RANK0_REMOTE_EID="${synthetic_rank0_remote_eid}"
    export A5_CCU_SYNTHETIC_DIE_ID="${synthetic_die}"
    export A5_CCU_SYNTHETIC_HOP="${synthetic_hop}"
else
    unset A5_CCU_SYNTHETIC_RANK0_LOCAL_EID A5_CCU_SYNTHETIC_RANK0_REMOTE_EID
    unset A5_CCU_SYNTHETIC_DIE_ID A5_CCU_SYNTHETIC_HOP
fi
if (( rebuild_public != 0 )); then
    export A5_CCU_REBUILD_PUBLIC_FIELDS=1
else
    unset A5_CCU_REBUILD_PUBLIC_FIELDS
fi
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH:-}"

(( skip_build != 0 )) || make -C "${test_dir}"

resolve_hccn_tool() {
    if [[ -n "${hccn_tool_path}" ]]; then
        [[ -x "${hccn_tool_path}" ]] || {
            echo "hccn_tool is not executable: ${hccn_tool_path}" >&2
            return 1
        }
        return 0
    fi
    if command -v hccn_tool >/dev/null 2>&1; then
        hccn_tool_path=$(command -v hccn_tool)
    elif [[ -x /usr/local/Ascend/driver/tools/hccn_tool ]]; then
        hccn_tool_path=/usr/local/Ascend/driver/tools/hccn_tool
    else
        echo "hccn_tool not found; pass --hccn-tool /path/to/hccn_tool" >&2
        return 1
    fi
}

declare -a hccn_targets=()

discover_hccn_targets() {
    local output_dir=$1
    local dev output_file
    local -a stat_devices
    hccn_targets=()
    IFS=',' read -ra stat_devices <<< "${hccn_devices}"
    for dev in "${stat_devices[@]}"; do
        dev=${dev//[[:space:]]/}
        [[ "${dev}" =~ ^[0-9]+$ ]] || {
            echo "invalid physical device ID in --hccn-devices: ${dev}" >&2
            return 1
        }
        output_file="${output_dir}/device_info_device${dev}.txt"
        {
            echo "# command: ${hccn_tool_path} -g -dev_info -i ${dev}"
            echo "# timestamp: $(date --iso-8601=ns)"
            env -u ASCEND_RT_VISIBLE_DEVICES -u ASCEND_VISIBLE_DEVICES \
                "${hccn_tool_path}" -g -dev_info -i "${dev}"
        } >"${output_file}" 2>&1 || true

        local found=0 udie port state
        while read -r udie port state; do
            [[ "$state" == "UP" ]] || continue
            hccn_targets+=("${dev}:${udie}:${port}")
            found=1
        done < <(awk -F'|' '
            /^\|[[:space:]]*[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+[[:space:]]*\|/ {
                gsub(/[[:space:]]/, "", $2); gsub(/[[:space:]]/, "", $3);
                gsub(/[[:space:]]/, "", $6); print $2, $3, $6
            }' "${output_file}")
        if (( found == 0 )); then
            echo "WARNING: no A5 UP ports discovered for device ${dev}; trying legacy statistics mode" >&2
            hccn_targets+=("${dev}::")
        fi
    done
    (( ${#hccn_targets[@]} > 0 ))
}

capture_hccn_stats() {
    local phase=$1
    local output_dir=$2
    local target dev udie port target_name output_file
    local before_supported_file="${output_dir}/before_supported_targets.txt"
    local phase_supported_file="${output_dir}/${phase}_supported_targets.txt"
    local success_count=0
    : >"${phase_supported_file}"
    for target in "${hccn_targets[@]}"; do
        IFS=':' read -r dev udie port <<<"$target"
        if [[ -n "$udie" ]]; then
            target_name="device${dev}_udie${udie}_port${port}"
        else
            target_name="device${dev}"
        fi
        if [[ "${phase}" == "after" ]] && ! grep -qx "${target}" "${before_supported_file}"; then
            continue
        fi
        output_file="${output_dir}/${phase}_${target_name}.txt"
        echo "HCCN snapshot ${phase}: ${target_name}"
        {
            echo "# timestamp: $(date --iso-8601=ns)"
            if [[ -n "$udie" ]]; then
                echo "# command: ${hccn_tool_path} -g -stat -i ${dev} -u ${udie} -p ${port}"
                env -u ASCEND_RT_VISIBLE_DEVICES -u ASCEND_VISIBLE_DEVICES \
                    "${hccn_tool_path}" -g -stat -i "${dev}" -u "${udie}" -p "${port}"
            else
                echo "# command: ${hccn_tool_path} -i ${dev} -stat -g"
                env -u ASCEND_RT_VISIBLE_DEVICES -u ASCEND_VISIBLE_DEVICES \
                    "${hccn_tool_path}" -i "${dev}" -stat -g
            fi
        } >"${output_file}" 2>&1 || {
            echo "WARNING: hccn_tool does not provide -stat data for ${target_name}:" >&2
            sed 's/^/  | /' "${output_file}" >&2
            continue
        }
        ((success_count += 1))
        echo "${target}" >>"${phase_supported_file}"
    done
    if (( success_count == 0 )); then
        echo "WARNING: no requested device returned hccn_tool -stat data" >&2
        return 1
    fi
}

report_hccn_delta() {
    local output_dir=$1
    python3 - "${output_dir}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
targets = [
    line.strip()
    for line in (root / "before_supported_targets.txt").read_text().splitlines()
    if line.strip()
]
after_targets = {
    item.strip()
    for item in (root / "after_supported_targets.txt").read_text().splitlines()
    if item.strip()
}
summary_fields = (
    "nic_tx_all_pkg_num",
    "nic_tx_all_oct_num",
    "nic_rx_all_pkg_num",
    "nic_rx_all_oct_num",
    "roce_new_pkt_rty_num",
)
line_re = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*([0-9]+)\s*$")

def load(path):
    counters = {}
    for line in path.read_text(errors="replace").splitlines():
        match = line_re.match(line)
        if match:
            counters[match.group(1)] = int(match.group(2))
    return counters

rows = []
print("HCCN counter deltas (after - before):")
for target in targets:
    if target not in after_targets:
        continue
    dev, udie, port = target.split(":")
    name = f"device{dev}" if not udie else f"device{dev}_udie{udie}_port{port}"
    before = load(root / f"before_{name}.txt")
    after = load(root / f"after_{name}.txt")
    common = sorted(before.keys() & after.keys())
    deltas = {key: after[key] - before[key] for key in common}
    shown = " ".join(
        f"{key}={deltas[key]}" if key in deltas else f"{key}=N/A"
        for key in summary_fields
    )
    location = f"physical_device={dev}"
    if udie:
        location += f" udie={udie} port={port}"
    print(f"  {location} {shown}")
    for key in common:
        rows.append((dev, udie, port, key, before[key], after[key], deltas[key]))

output = root / "hccn_counter_deltas.tsv"
with output.open("w") as handle:
    handle.write("physical_device\tudie\tport\tcounter\tbefore\tafter\tdelta\n")
    for row in rows:
        handle.write("\t".join(map(str, row)) + "\n")
print(f"HCCN raw snapshots and full delta table: {root}")
PY
}

build_command() {
    local payload_bytes=$1
    local worker_rank=$2
    local root_info_file=$3
    command=()
    if [[ -n "${worker_preload}" ]]; then
        [[ -f "${worker_preload}" ]] || {
            echo "worker preload library not found: ${worker_preload}" >&2
            return 1
        }
        command+=(env "LD_PRELOAD=${worker_preload}")
        [[ -z "${worker_trace_prefix}" ]] || \
            command+=("A5_URMA_TP_TRACE_PREFIX=${worker_trace_prefix}.rank${worker_rank}"
                "A5_IOCTL_PAYLOAD_TRACE_PREFIX=${worker_trace_prefix}.rank${worker_rank}")
    fi
    command+=("${test_dir}/a5_ccu_urma_route_probe_test"
        --bytes "${payload_bytes}"
        --warmup "${warmup}"
        --iters "${iterations}"
        --route-index "${route_index}"
        --worker-rank "${worker_rank}"
        --root-info-file "${root_info_file}")
    [[ -z "${route_indices}" ]] || command+=(--route-indices "${route_indices}")
    (( remote_only == 0 )) || command+=(--remote-only)
    (( channel_only == 0 )) || command+=(--channel-only)
    (( comm_init_only == 0 )) || command+=(--comm-init-only)
}

run_pair() {
    local payload_bytes=$1
    local pair_state_dir
    pair_state_dir=$(mktemp -d /tmp/a5_ccu_urma_pair.XXXXXX)
    local root_info_file="${pair_state_dir}/root_info.bin"
    local status0=0
    local status1=0

    build_command "${payload_bytes}" 0 "${root_info_file}"
    local rank0_command=("${command[@]}")
    build_command "${payload_bytes}" 1 "${root_info_file}"
    local rank1_command=("${command[@]}")

    if (( profile == 0 )); then
        "${rank0_command[@]}" &
        local rank0_pid=$!
        "${rank1_command[@]}" &
        local rank1_pid=$!
        wait "${rank0_pid}" || status0=$?
        wait "${rank1_pid}" || status1=$?
    else
        command -v msprof >/dev/null || { echo "msprof not found in PATH" >&2; return 1; }
        local run_dir="${profile_root}/a5_ccu_urma_routes_${route_indices:-${route_index}}_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "${run_dir}/rank0" "${run_dir}/rank1"
        msprof --output="${run_dir}/rank0" --ascendcl=on --runtime-api=on \
            --task-time=l2 --hccl=on --type=text "${rank0_command[@]}" &
        local rank0_pid=$!
        msprof --output="${run_dir}/rank1" --ascendcl=on --runtime-api=on \
            --task-time=l2 --hccl=on --type=text "${rank1_command[@]}" &
        local rank1_pid=$!
        wait "${rank0_pid}" || status0=$?
        wait "${rank1_pid}" || status1=$?
        if (( status0 == 0 && status1 == 0 )); then
            mapfile -d '' prof_dirs < <(find "${run_dir}" -type d -name 'PROF_*' -print0 2>/dev/null)
            (( ${#prof_dirs[@]} > 0 )) || { echo "No PROF_* directory under ${run_dir}" >&2; return 1; }
            for prof_dir in "${prof_dirs[@]}"; do
                msprof --export=on --output="${prof_dir}"
            done
            echo "Profile output: ${run_dir}"
            find "${run_dir}" -type f -name '*.csv' | sort
        fi
    fi
    find "${pair_state_dir}" -type f -delete
    rmdir "${pair_state_dir}"
    (( status0 == 0 && status1 == 0 )) || {
        echo "rank workers failed: rank0=${status0}, rank1=${status1}" >&2
        return 1
    }
}

run_workload() {
if (( sweep != 0 )); then
    echo "Physical devices : ${ASCEND_RT_VISIBLE_DEVICES}"
    echo "Selected routes : ${route_indices:-${A5_CCU_ROUTE_INDEX}}"
    echo "Mode            : $([[ ${remote_only} -eq 1 ]] && echo remote-only || echo allgather)"
    for sweep_bytes in 65536 262144 1048576 2097152 8388608 33554432; do
        echo "===== payload ${sweep_bytes} bytes ====="
        run_pair "${sweep_bytes}"
    done
    return 0
fi

echo "Physical devices : ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Selected routes : ${route_indices:-${A5_CCU_ROUTE_INDEX}}"
echo "Payload/rank    : ${bytes} bytes"
echo "Mode            : $([[ ${channel_only} -eq 1 ]] && echo channel-only || \
    ([[ ${comm_init_only} -eq 1 ]] && echo comm-init-only || \
    ([[ ${remote_only} -eq 1 ]] && echo remote-only || echo allgather)))"

run_pair "${bytes}"
}

if (( hccn_stat == 0 )); then
    run_workload
    exit $?
fi

resolve_hccn_tool
[[ -n "${hccn_stat_root}" ]] || hccn_stat_root="${profile_root}"
route_label=${route_indices:-${route_index}}
route_label=${route_label//[^[:alnum:]_-]/_}
hccn_run_dir="${hccn_stat_root}/hccn_routes_${route_label}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "${hccn_run_dir}"

hccn_available=1
discover_hccn_targets "${hccn_run_dir}" || hccn_available=0
capture_hccn_stats before "${hccn_run_dir}" || hccn_available=0
workload_status=0
run_workload || workload_status=$?
after_status=0
if (( hccn_available != 0 )); then
    capture_hccn_stats after "${hccn_run_dir}" || after_status=$?
fi
if (( hccn_available != 0 && after_status == 0 )); then
    report_hccn_delta "${hccn_run_dir}"
elif (( hccn_available == 0 )); then
    echo "HCCN statistics unavailable; workload result is still preserved." >&2
fi
(( workload_status == 0 )) || exit "${workload_status}"
exit "${after_status}"
