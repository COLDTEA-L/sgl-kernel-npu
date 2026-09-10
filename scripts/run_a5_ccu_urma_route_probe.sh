#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
test_dir="${repo_root}/examples/a5_ccu_urma_route_probe/testcase"

devices="2,3"
route_index=0
bytes=2097152
warmup=10
iterations=100
profile=0
profile_root=/home/l00934901/profiling

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) devices=$2; shift 2 ;;
        --route-index) route_index=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --iters) iterations=$2; shift 2 ;;
        --profile) profile=1; shift ;;
        --profile-root) profile_root=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

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
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH:-}"

make -C "${test_dir}"
command=("${test_dir}/a5_ccu_urma_route_probe_test"
    --bytes "${bytes}"
    --warmup "${warmup}"
    --iters "${iterations}"
    --route-index "${route_index}")

echo "Physical devices : ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Selected route  : ${A5_CCU_ROUTE_INDEX}"
echo "Payload/rank    : ${bytes} bytes"

if (( profile == 0 )); then
    "${command[@]}"
    exit 0
fi

command -v msprof >/dev/null || { echo "msprof not found in PATH" >&2; exit 1; }
run_dir="${profile_root}/a5_ccu_urma_route_${route_index}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${run_dir}"
msprof \
    --output="${run_dir}" \
    --ascendcl=on \
    --runtime-api=on \
    --task-time=l2 \
    --hccl=on \
    --type=text \
    "${command[@]}"

mapfile -d '' prof_dirs < <(find "${run_dir}" -type d -name 'PROF_*' -print0 2>/dev/null)
(( ${#prof_dirs[@]} > 0 )) || { echo "No PROF_* directory under ${run_dir}" >&2; exit 1; }
for prof_dir in "${prof_dirs[@]}"; do
    msprof --export=on --output="${prof_dir}"
done
echo "Profile output: ${run_dir}"
find "${run_dir}" -type f -name '*.csv' | sort
