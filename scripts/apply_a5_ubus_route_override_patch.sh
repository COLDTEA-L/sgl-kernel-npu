#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
kernel_root=""
apply=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel-root) kernel_root=$2; shift 2 ;;
        --apply) apply=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "${kernel_root}" ]] || { echo "--kernel-root is required" >&2; exit 2; }
[[ -d "${kernel_root}/drivers/ub/ubus" ]] || {
    echo "not an openEuler kernel source tree: ${kernel_root}" >&2
    exit 2
}
patch_file="${repo_root}/kernel_patches/openeuler-6.6/0001-a5-ubus-route-override.patch"

echo "Running kernel : $(uname -r)"
echo "Kernel source  : ${kernel_root}"
echo "Patch          : ${patch_file}"
git -C "${kernel_root}" status --short --branch
git -C "${kernel_root}" apply --check "${patch_file}"
if (( apply == 0 )); then
    echo "Patch check passed. Re-run with --apply to modify the kernel source tree."
    exit 0
fi
git -C "${kernel_root}" apply "${patch_file}"
echo "Patch applied. Review with: git -C ${kernel_root} diff -- drivers/ub/ubus"
