#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${A5_BUILD_PYTHON_ENV:-}"

usage()
{
    cat <<'EOF'
Usage: bash scripts/build_a5_notify_dispatch.sh [--python-env PATH]

Build a DeepEP wheel containing only the A5 NotifyDispatch custom operator.
NotifyDispatch obtains the HCCL context and communication-window base address.
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

if [[ -z "${PYTHON_ENV}" ]] \
    && ! python3 -c 'import importlib.util; assert importlib.util.find_spec("torch") and importlib.util.find_spec("torch_npu") and importlib.util.find_spec("pybind11")' >/dev/null 2>&1 \
    && [[ -x /opt/conda/envs/cam_py311_pt28/bin/python3 ]]; then
    PYTHON_ENV=/opt/conda/envs/cam_py311_pt28
fi

if [[ -n "${PYTHON_ENV}" ]]; then
    [[ -x "${PYTHON_ENV}/bin/python3" ]] || {
        echo "Invalid Python environment: ${PYTHON_ENV}" >&2
        exit 1
    }
    export PATH="${PYTHON_ENV}/bin:${PATH}"
fi

export DEEPEP_SINGLE_OP=notify_dispatch
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
notify = [name for name in names if "/kernel/ascend950/notify_dispatch/" in name and name.endswith(".o")]
other = [
    name
    for name in names
    if "/kernel/ascend950/" in name and name.endswith(".o") and "/notify_dispatch/" not in name
]
if not notify:
    raise SystemExit("wheel does not contain the NotifyDispatch device binary")
if other:
    raise SystemExit(f"wheel unexpectedly contains other device binaries: {other[:5]}")
print(f"NotifyDispatch device binaries: {len(notify)}")
PY

echo "Built: ${wheel}"
sha256sum "${wheel}"
echo "Total build time: ${elapsed}s"
