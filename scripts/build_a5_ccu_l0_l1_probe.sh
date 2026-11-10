#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: bash $0 CANN_PATH BUILD_DIR" >&2
  exit 2
fi
A5_CANN=$(realpath "$1")
A5_BUILD=$2
A5_ROOT=$(cd "$(dirname "$0")/.." && pwd)
set +u
source "$A5_CANN/set_env.sh"
set -u
mkdir -p "$A5_BUILD"
cmake -S "$A5_ROOT/examples/a5_ccu_l0_l1_probe" -B "$A5_BUILD" \
  -DASCEND_CANN_PACKAGE_PATH="$A5_CANN" -DCMAKE_BUILD_TYPE=Release \
  2>&1 | tee "$A5_BUILD/configure.log"
cmake --build "$A5_BUILD" --parallel 4 2>&1 | tee "$A5_BUILD/build.log"
nm -D -C --undefined-only "$A5_BUILD/a5_ccu_l0_l1_probe" > "$A5_BUILD/undefined.txt"
if grep -Eq 'hcomm::CcuRep::|HcclCcuKernel|HcclCcuUrma' "$A5_BUILD/undefined.txt"; then
  echo "L0L1_BUILD_FAILED legacy or route dependency" >&2; exit 1
fi
"$A5_BUILD/a5_ccu_l0_l1_probe" --check-abi | tee "$A5_BUILD/abi.log"
echo "L0L1_BUILD_PASS binary=$A5_BUILD/a5_ccu_l0_l1_probe"
