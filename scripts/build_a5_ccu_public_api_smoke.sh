#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: bash $0 /usr/local/Ascend/cann-9.3.0 /home/l00934901/profiling/ccu-public-build" >&2
  exit 2
fi
A5_CANN=$(realpath "$1")
A5_BUILD=$2
A5_ROOT=$(cd "$(dirname "$0")/.." && pwd)
[[ -f "$A5_CANN/set_env.sh" ]]
# Use only the explicitly selected installation; do not source the cann alias.
set +u
source "$A5_CANN/set_env.sh"
set -u
mkdir -p "$A5_BUILD"
cmake -S "$A5_ROOT/examples/a5_ccu_public_api_smoke" -B "$A5_BUILD" \
  -DASCEND_CANN_PACKAGE_PATH="$A5_CANN" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  2>&1 | tee "$A5_BUILD/configure.log"
cmake --build "$A5_BUILD" --parallel 4 2>&1 | tee "$A5_BUILD/build.log"
nm -D -C --undefined-only "$A5_BUILD/a5_ccu_public_api_smoke" > "$A5_BUILD/undefined.txt"
if grep -Eq 'hcomm::CcuRep::|HcclCcuKernel' "$A5_BUILD/undefined.txt"; then
  echo "PUBLIC_CCU_BUILD_FAILED legacy symbol dependency" >&2
  exit 1
fi
echo "PUBLIC_CCU_BUILD_PASS binary=$A5_BUILD/a5_ccu_public_api_smoke"
