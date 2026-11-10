#!/usr/bin/env bash
set -euo pipefail
if (( $# != 2 )); then
    echo "Usage: bash $0 CANN_ROOT BUILD_DIR" >&2
    exit 2
fi
cann_root=$(realpath "$1")
build_dir=$2
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ -f "$cann_root/include/acl/acl_rt.h" && -f "$cann_root/set_env.sh" ]] || {
    echo "Missing SDK headers/set_env.sh under $cann_root" >&2; exit 2;
}
mkdir -p "$build_dir"
build_dir=$(cd "$build_dir" && pwd)
# Use the explicitly selected SDK; avoid the legacy patched-HCCL preload.
unset LD_PRELOAD
set +u
source "$cann_root/set_env.sh"
set -u
export LD_LIBRARY_PATH="$cann_root/lib64:${LD_LIBRARY_PATH:-}"
"${CXX:-c++}" -std=c++14 -Wall -Wextra -Werror \
    -I"$cann_root/include" \
    "$repo_root/examples/a5_ccu_public_api_smoke/valuewait_runtime_abi.cc" \
    -L"$cann_root/lib64" -Wl,-rpath,"$cann_root/lib64" \
    -Wl,-z,defs -lascendcl -ldl \
    -o "$build_dir/a5_valuewait_runtime_abi"
"$build_dir/a5_valuewait_runtime_abi"
