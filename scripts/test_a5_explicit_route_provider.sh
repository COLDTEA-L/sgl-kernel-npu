#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

cann_root=${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is not set}
test_dir="${repo_root}/examples/a5_ccu_urma_route_probe/testcase"
build_dir=$(mktemp -d /tmp/a5_route_provider_smoke.XXXXXX)
trap 'rm -rf "${build_dir}"' EXIT

c++ -std=c++14 -shared -fPIC \
    -I"${cann_root}/include" \
    -I"${repo_root}/examples/a5_ccu_urma_route_probe/inc" \
    "${test_dir}/mock_explicit_route_backend.cc" \
    -o "${build_dir}/libmock_a5_uvs_route_backend.so"

c++ -std=c++14 \
    -I"${cann_root}/include" \
    -I"${repo_root}/examples/a5_ccu_urma_route_probe/inc" \
    "${test_dir}/explicit_route_provider_smoke.cc" -ldl \
    -o "${build_dir}/explicit_route_provider_smoke"

python3 "${repo_root}/scripts/probe_a5_urma_source_route_capability.py" \
    --umdk-root "${UMDK_REPO:-/home/liuyuanwen/umdk}" \
    --backend "${build_dir}/libmock_a5_uvs_route_backend.so" \
    --output "${build_dir}/capability.json"

provider="${cann_root}/opp/vendors/cust/lib64/liba5_uvs_manifest_route_provider.so"
[[ -f "${provider}" ]] || {
    echo "provider is not installed: ${provider}" >&2
    exit 2
}
"${build_dir}/explicit_route_provider_smoke" \
    "${provider}" \
    "${build_dir}/libmock_a5_uvs_route_backend.so" \
    "${repo_root}/examples/a5_ccu_urma_route_probe/config/explicit_relay_example.csv"
