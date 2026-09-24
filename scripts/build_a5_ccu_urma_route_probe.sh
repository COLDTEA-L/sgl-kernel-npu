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
bridge_path=""
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

cleanup()
{
    if [[ -n "${bridge_path}" && -d "${bridge_path}" ]]; then
        rm -rf -- "${bridge_path}"
    fi
    if [[ -n "${temporary_cann_cmake_link}" && -L "${temporary_cann_cmake_link}" ]]; then
        rm -f -- "${temporary_cann_cmake_link}"
    fi
}
trap cleanup EXIT

# Some secured server images apply different transparent-encryption policies to
# the HCCL and sgl-kernel-npu workspaces.  Copying text sources into a temporary
# directory below HCCL can therefore expose ciphertext to cmake.  HCCL invokes
# add_subdirectory(CUSTOM_OPS_PATH) without a binary directory, so the custom-op
# tree must still appear below HCCL.  Make an in-tree hard-link farm: no source
# bytes are copied, while CMake still sees an ordinary in-tree directory.
project_cmake="${project_dir}/CMakeLists.txt"
[[ -s "${project_cmake}" ]] || {
    echo "Missing custom-op CMakeLists.txt: ${project_cmake}" >&2
    exit 1
}
first_byte=$(LC_ALL=C head -c 1 "${project_cmake}" || true)
if [[ "${first_byte}" != "#" && "${first_byte}" != "c" ]]; then
    echo "Custom-op CMakeLists.txt is not readable text: ${project_cmake}" >&2
    file "${project_cmake}" >&2 || true
    od -An -tx1 -N32 "${project_cmake}" >&2 || true
    exit 1
fi
bridge_path=$(mktemp -d "${hccl_repo}/.a5_ccu_urma_route_probe.XXXXXX")
if ! cp -al "${project_dir}/." "${bridge_path}/"; then
    echo "Failed to hard-link the custom-op tree into ${hccl_repo}." >&2
    echo "The HCCL and sgl-kernel-npu repositories must be on the same filesystem." >&2
    exit 1
fi

bridge_cmake="${bridge_path}/CMakeLists.txt"
if ! cmp -s "${project_cmake}" "${bridge_cmake}"; then
    echo "Custom-op staging changed CMakeLists.txt contents: ${bridge_cmake}" >&2
    file "${project_cmake}" "${bridge_cmake}" >&2 || true
    od -An -tx1 -N32 "${project_cmake}" >&2 || true
    od -An -tx1 -N32 "${bridge_cmake}" >&2 || true
    exit 1
fi

source_inode=$(stat -c '%d:%i' "${project_cmake}")
bridge_inode=$(stat -c '%d:%i' "${bridge_cmake}")
if [[ "${source_inode}" != "${bridge_inode}" ]]; then
    echo "Custom-op staging is not a hard link: source=${source_inode} staged=${bridge_inode}" >&2
    exit 1
fi

bridge_relative=${bridge_path#"${hccl_repo}/"}
echo "Custom ops source : ${project_dir}"
echo "Hard-link staging : ${bridge_path}"

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
