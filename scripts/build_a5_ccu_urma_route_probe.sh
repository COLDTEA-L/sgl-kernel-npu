#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
project_dir="${repo_root}/examples/a5_ccu_urma_route_probe"
install_after_build=0
install_path=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) install_after_build=1; shift ;;
        --install-path) install_path=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -n "${HCCL_REPO:-}" ]]; then
    hccl_repo=${HCCL_REPO}
elif [[ -x /home/l00934901/hccl/build.sh ]]; then
    hccl_repo=/home/l00934901/hccl
else
    hccl_repo=/home/liuyuanwen/hccl
fi

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

temporary_cann_cmake_link=""
cann_cmake_dir="${hccl_repo}/third_party/cann-cmake"
if [[ ! -f "${cann_cmake_dir}/function/prepare.cmake" ]]; then
    for cached_cann_cmake in \
        "${hccl_repo}/build_device/_deps/cann-cmake-src" \
        "${hccl_repo}/build/_deps/cann-cmake-src"
    do
        if [[ -f "${cached_cann_cmake}/function/prepare.cmake" ]]; then
            if [[ -e "${cann_cmake_dir}" || -L "${cann_cmake_dir}" ]]; then
                echo "Invalid incomplete cann-cmake cache: ${cann_cmake_dir}" >&2
                echo "Move that path aside, or populate it with a complete cann-cmake checkout." >&2
                exit 1
            fi
            mkdir -p "${hccl_repo}/third_party"
            ln -s "${cached_cann_cmake}" "${cann_cmake_dir}"
            temporary_cann_cmake_link="${cann_cmake_dir}"
            echo "Offline cann-cmake: ${cached_cann_cmake}"
            break
        fi
    done
fi

bridge_path=$(mktemp -d "${hccl_repo}/.a5_ccu_urma_route_probe.XXXXXX")
cleanup()
{
    rm -rf -- "${bridge_path}"
    if [[ -n "${temporary_cann_cmake_link}" && -L "${temporary_cann_cmake_link}" ]]; then
        rm -f -- "${temporary_cann_cmake_link}"
    fi
}
trap cleanup EXIT
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

if (( install_after_build == 0 )); then
    echo "Run this script again with --install to install into the active CANN tree."
    exit 0
fi

if [[ -z "${install_path}" ]]; then
    [[ -n "${ASCEND_HOME_PATH:-}" ]] || {
        echo "ASCEND_HOME_PATH is not set; pass --install-path explicitly." >&2
        exit 1
    }
    install_path=$(readlink -f "${ASCEND_HOME_PATH}")
fi
[[ -d "${install_path}/opp" ]] || {
    echo "Invalid CANN install path: ${install_path}" >&2
    exit 1
}

echo "Installing into CANN: ${install_path}"
env -u ASCEND_CUSTOM_OPP_PATH -u ASCEND_OPP_PATH \
    "${latest_package}" --quiet --install "--install-path=${install_path}"

test -f "${install_path}/opp/vendors/cust/include/a5_ccu_urma_route_probe.h"
test -f "${install_path}/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so"
echo "Installed header: ${install_path}/opp/vendors/cust/include/a5_ccu_urma_route_probe.h"
echo "Installed library: ${install_path}/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so"
