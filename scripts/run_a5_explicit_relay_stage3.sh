#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

devices="6,7"
relay_device="2"
manifest="${repo_root}/examples/a5_ccu_urma_route_probe/config/explicit_relay_example.csv"
backend="${A5_UVS_EXPLICIT_ROUTE_BACKEND:-}"
bytes=4194304
warmup=10
iterations=100
channel_only=0
hccn_stat=0
hccn_stat_root=/home/l00934901/profiling

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) devices=$2; shift 2 ;;
        --relay-device) relay_device=$2; shift 2 ;;
        --manifest) manifest=$2; shift 2 ;;
        --backend) backend=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --iters) iterations=$2; shift 2 ;;
        --channel-only) channel_only=1; shift ;;
        --hccn-stat) hccn_stat=1; shift ;;
        --hccn-stat-root) hccn_stat_root=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

IFS=',' read -r src_device dst_device extra <<<"${devices}"
[[ -n "${src_device}" && -n "${dst_device}" && -z "${extra:-}" ]] || {
    echo "--devices must contain exactly two physical device IDs" >&2
    exit 2
}
[[ "${relay_device}" != "${src_device}" && "${relay_device}" != "${dst_device}" ]] || {
    echo "relay device must differ from both endpoint devices" >&2
    exit 2
}
[[ -f "${manifest}" ]] || { echo "manifest not found: ${manifest}" >&2; exit 2; }
awk -F, -v relay="${relay_device}" '
    $1 !~ /^#/ && $1 != "route_id" && $6 == relay { found=1 }
    END { exit(found ? 0 : 1) }
' "${manifest}" || {
    echo "manifest does not contain relay_phy=${relay_device}" >&2
    exit 2
}
[[ -n "${backend}" ]] || {
    echo "--backend or A5_UVS_EXPLICIT_ROUTE_BACKEND is required" >&2
    exit 2
}

python3 "${repo_root}/scripts/probe_a5_urma_source_route_capability.py" \
    --umdk-root "${UMDK_REPO:-/home/l00934901/umdk}" \
    --backend "${backend}" \
    --output "${hccn_stat_root}/a5_explicit_relay_capability.json"

provider="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/opp/vendors/cust/lib64/"\
"liba5_uvs_manifest_route_provider.so"
[[ -f "${provider}" ]] || {
    echo "manifest route provider is not installed: ${provider}" >&2
    exit 2
}

export A5_UVS_EXPLICIT_ROUTE_BACKEND="${backend}"
command=(bash "${repo_root}/scripts/run_a5_ccu_urma_route_probe.sh"
    --devices "${devices}"
    --source-route-manifest "${manifest}"
    --source-route-provider "${provider}"
    --bytes "${bytes}"
    --warmup "${warmup}"
    --iters "${iterations}")
if (( channel_only == 0 )); then
    command+=(--remote-only)
else
    command+=(--channel-only)
fi
if (( hccn_stat != 0 )); then
    command+=(--hccn-stat --hccn-devices "${src_device},${dst_device},${relay_device}"
              --hccn-stat-root "${hccn_stat_root}")
fi
"${command[@]}"
