#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${A5_BUILD_PYTHON_ENV:-}"

usage()
{
    cat <<'EOF'
Usage: bash scripts/build_a5_io_die_all2all_detour.sh [--python-env PATH]

Build a focused DeepEP wheel for the two A5 AllToAll validation stages:
  1. fixed-count HCCL AllToAll using the CCU runtime;
  2. AIV+URMA two-hop AllToAll using explicit relay-rank windows and
     ping-pong CpGM2GM copies inside each path block.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python-env)
            [[ $# -ge 2 ]] || { echo "--python-env requires a path" >&2; exit 1; }
            PYTHON_ENV="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -n "${PYTHON_ENV}" ]]; then
    [[ -x "${PYTHON_ENV}/bin/python3" ]] || {
        echo "Invalid Python environment: ${PYTHON_ENV}" >&2
        exit 1
    }
    export PATH="${PYTHON_ENV}/bin:${PATH}"
fi

python3 -c 'import importlib.util; assert importlib.util.find_spec("torch") and importlib.util.find_spec("torch_npu") and importlib.util.find_spec("pybind11")' >/dev/null 2>&1 || {
    echo "python3 must provide torch, torch_npu and pybind11" >&2
    exit 1
}

export DEEPEP_SINGLE_OP=a5_all2all_validation
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

cd "${PROJECT_ROOT}"
start_time="$(date +%s)"
bash build.sh -a deepep Ascend950
elapsed="$(( $(date +%s) - start_time ))"

wheel="$(ls -1t output/deep_ep*.whl | head -1)"
python3 - "${wheel}" <<'PY'
import sys
import zipfile

wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
ccu = [
    name for name in names
    if "/kernel/ascend950/hccl_all2_all_ccu/" in name and name.endswith(".o")
]
detour = [
    name for name in names
    if "/kernel/ascend950/all2_all_detour_io_die/" in name and name.endswith(".o")
]
if not ccu:
    raise SystemExit("wheel does not contain the HcclAll2AllCcu device binary")
if not detour:
    raise SystemExit("wheel does not contain the All2AllDetourIoDie device binary")
print(f"HcclAll2AllCcu device binaries: {len(ccu)}")
print(f"All2AllDetourIoDie device binaries: {len(detour)}")
PY

echo "Built: ${wheel}"
sha256sum "${wheel}"
echo "Total build time: ${elapsed}s"
