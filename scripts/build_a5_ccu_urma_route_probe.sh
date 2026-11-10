#!/usr/bin/env bash
# CANN 9.2/9.3 branch: standalone public SDK build, no HCCL source checkout.
set -Eeuo pipefail
A5_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
A5_CANN=${A5_CANN_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.3.0}}
A5_INSTALL=0
A5_PREFIX=""
A5_BUILD=${A5_BUILD_DIR:-/home/l00934901/profiling/a5-public-route-build}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install) A5_INSTALL=1; shift ;;
    --install-path) A5_PREFIX=$2; shift 2 ;;
    --cann-path) A5_CANN=$2; shift 2 ;;
    --build-dir) A5_BUILD=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
A5_CANN=$(realpath "$A5_CANN")
[[ -f "$A5_CANN/include/hcomm/ccu/ccu_launch.h" ]] || {
  echo "Public CCU headers missing: $A5_CANN; use the original branch for 9.1" >&2; exit 1;
}
set +u
source "$A5_CANN/set_env.sh"
set -u
A5_PREFIX=${A5_PREFIX:-$A5_CANN}
mkdir -p "$A5_BUILD"
cmake -S "$A5_REPO/examples/a5_ccu_urma_route_probe" -B "$A5_BUILD" \
  -DASCEND_CANN_PACKAGE_PATH="$A5_CANN" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DA5_CCU_BUILD_HOST_TESTS=ON \
  -DCUSTOM_OPS_OPP_LIB_PATH=opp/vendors/cust/lib64 \
  -DCUSTOM_OPS_OPP_INC_PATH=opp/vendors/cust/include \
  2>&1 | tee "$A5_BUILD/configure.log"
cmake --build "$A5_BUILD" --parallel 4 2>&1 | tee "$A5_BUILD/build.log"
ctest --test-dir "$A5_BUILD" --output-on-failure 2>&1 | tee "$A5_BUILD/host-test.log"
A5_LIB="$A5_BUILD/liba5_ccu_urma_route_probe.so"
nm -D -C --undefined-only "$A5_LIB" > "$A5_BUILD/undefined.txt"
if grep -Eq 'hcomm::CcuRep::|hcomm::CcuKernel|HcclCcuKernel|HcclDfxRegOpInfoByCommId' "$A5_BUILD/undefined.txt"; then
  echo "PUBLIC_ROUTE_BUILD_FAILED legacy dependency" >&2; exit 1
fi
python3 - "$A5_LIB" <<'CHECK_ABI'
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
for name, expected in [('A5CcuPublicApiAbiVersion', 1), ('A5CcuUrmaPreparedPlanAbiVersion', 16),
                       ('A5CcuPeerPlanAbiVersion', 5), ('A5CcuAllToAllVAbiVersion', 1),
                       ('A5CcuAllToAllVLayoutAbiVersion', 1), ('A5CcuEp8DfxAbiVersion', 2),
                       ('A5CcuEidAllToAllVAbiVersion', 1),
                       ('A5CcuEidV1CompatAbiVersion', 1),
                       ('A5CcuEidPeerResourceAbiVersion', 1),
                       ('A5CcuEidRegisterProbeAbiVersion', 1), ('A5CcuEidDirectControlAbiVersion', 1), ('A5CcuEidCompareProbeAbiVersion', 1)]:
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int
    actual = fn()
    if actual != expected: raise RuntimeError(f'{name}: expected {expected}, got {actual}')
    print(f'PUBLIC_ROUTE_ABI {name}={actual}')
print('PUBLIC_ROUTE_LOAD_PASS')
CHECK_ABI
if [[ "$A5_INSTALL" == 1 ]]; then
  cmake --install "$A5_BUILD" --prefix "$A5_PREFIX" --component hccl
fi
echo "PUBLIC_ROUTE_BUILD_PASS library=$A5_LIB"
