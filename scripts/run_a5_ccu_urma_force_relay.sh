#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
src_phy=""; relay_phy=""; dst_phy=""; phy_map=""
bytes=4194304; warmup=10; iterations=100; hccn_stat=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-phy) src_phy=$2; shift 2 ;;
        --relay-phy) relay_phy=$2; shift 2 ;;
        --dst-phy) dst_phy=$2; shift 2 ;;
        --phy-map) phy_map=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --iters) iterations=$2; shift 2 ;;
        --hccn-stat) hccn_stat=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "${src_phy}" && -n "${relay_phy}" && -n "${dst_phy}" && -n "${phy_map}" ]] || {
    echo "--src-phy, --relay-phy, --dst-phy and --phy-map are required" >&2
    exit 2
}

ctl=(python3 "${repo_root}/scripts/a5_ub_route_ctl.py"
    --phy-map "${phy_map}" --src-phy "${src_phy}"
    --relay-phy "${relay_phy}" --dst-phy "${dst_phy}")

cleanup() {
    "${ctl[@]}" clear || echo "WARNING: automatic UBUS route restore failed" >&2
}
trap cleanup EXIT INT TERM

"${ctl[@]}" resolve
"${ctl[@]}" force

command=(bash "${repo_root}/scripts/run_a5_ccu_urma_route_probe.sh"
    --devices "${src_phy},${dst_phy}" --route-index 0
    --one-way-src-rank 0 --bytes "${bytes}" --warmup "${warmup}" --iters "${iterations}")
if (( hccn_stat != 0 )); then
    command+=(--hccn-stat --hccn-devices "${src_phy},${relay_phy},${dst_phy}"
              --hccn-stat-root /home/l00934901/profiling)
fi
"${command[@]}"
