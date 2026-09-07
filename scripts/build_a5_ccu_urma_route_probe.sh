#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
project_dir="${repo_root}/examples/a5_ccu_urma_route_probe"
hccl_repo=${HCCL_REPO:-/home/liuyuanwen/hccl}

if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

[[ -x "${hccl_repo}/build.sh" ]] || {
    echo "HCCL build.sh not found: ${hccl_repo}/build.sh" >&2
    echo "Set HCCL_REPO to the HCCL source directory." >&2
    exit 1
}

echo "CANN home : ${ASCEND_HOME_PATH:-not-set}"
echo "HCCL repo : ${hccl_repo}"
echo "Probe src : ${project_dir}"

bridge_path=$(mktemp -d "${hccl_repo}/.a5_ccu_urma_route_probe.XXXXXX")
trap 'rm -rf -- "${bridge_path}"' EXIT
cp -a "${project_dir}/." "${bridge_path}/"
bridge_relative=${bridge_path#"${hccl_repo}/"}

cd "${hccl_repo}"
bash build.sh \
    --vendor=cust \
    --ops=ccu_urma_route_probe \
    --custom_ops_path="${bridge_relative}"

latest_package=$(
    find "${hccl_repo}/build_out" -maxdepth 1 -type f \
        -name '*ccu_urma_route_probe*.run' -printf '%T@ %p\n' 2>/dev/null |
        sort -nr | head -n 1 | cut -d' ' -f2-
)
[[ -n "${latest_package}" ]] || {
    echo "Build completed but no route-probe .run package was found." >&2
    exit 1
}

echo "Build succeeded. Latest package:"
echo "${latest_package}"
