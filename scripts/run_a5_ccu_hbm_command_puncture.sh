#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
src_phy=""
dst_phy=""
relay_phys=""
relay_planes=""
direct_route=0
bytes=4194304
timeout_seconds=600
output_root=/home/l00934901/profiling
topology="${repo_root}/docs/topology/a5_hccn_device_topology_raw.txt"
topology_json=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
cann_root=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.T560}
graph=1
worker_register_mode=full
register_only=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-phy) src_phy=$2; shift 2 ;;
        --dst-phy) dst_phy=$2; shift 2 ;;
        --relay-phys) relay_phys=$2; shift 2 ;;
        --relay-planes) relay_planes=$2; shift 2 ;;
        --direct-route) direct_route=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --timeout-seconds) timeout_seconds=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --topology) topology=$2; shift 2 ;;
        --topology-json) topology_json=$2; shift 2 ;;
        --cann-root) cann_root=$2; shift 2 ;;
        --no-graph) graph=0; shift ;;
        --worker-register-mode) worker_register_mode=$2; shift 2 ;;
        --register-only) register_only=1; graph=0; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "${worker_register_mode}" in
    full|hbm_once|loop_only|loop_hbm) ;;
    *) echo "--worker-register-mode must be full, hbm_once, loop_only, or loop_hbm" >&2; exit 2 ;;
esac

[[ "${src_phy}" =~ ^[0-9]+$ && "${dst_phy}" =~ ^[0-9]+$ ]] || {
    echo "--src-phy and --dst-phy are required" >&2; exit 2;
}
IFS=',' read -ra relay_array <<<"${relay_phys}"
(( ${#relay_array[@]} >= 1 && ${#relay_array[@]} <= 7 )) || {
    echo "--relay-phys requires 1..7 ordered physical relay cards" >&2; exit 2;
}

cd "${repo_root}"
source "${cann_root}/set_env.sh"
set +u
source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash
set -u
unset LD_PRELOAD
export ASCEND_RT_VISIBLE_DEVICES="${src_phy},${dst_phy}"
export HCCL_OP_EXPANSION_MODE=CCU_SCHED

route_so=${A5_CCU_ROUTE_PROBE_LIB:-${cann_root}/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so}
[[ -f "${route_so}" ]] || { echo "missing route probe: ${route_so}" >&2; exit 2; }
for symbol in HcclCcuUrmaCommandBlockWorkerCreate HcclCcuUrmaCommandBlockWorkerStop A5CcuHbmCommandPunctureAbiVersion; do
    nm -D "${route_so}" | grep " ${symbol}$" >/dev/null || {
        echo "stale route probe: ${symbol} is missing" >&2; exit 2;
    }
done
export A5_CCU_ROUTE_PROBE_LIB=$(readlink -f "${route_so}")
export LD_LIBRARY_PATH="$(dirname "${A5_CCU_ROUTE_PROBE_LIB}"):${LD_LIBRARY_PATH}"

python3 - "${A5_CCU_ROUTE_PROBE_LIB}" <<'PY'
import ctypes
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve()
lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
abi = lib.A5CcuHbmCommandPunctureAbiVersion
abi.restype = ctypes.c_int
print("Route puncture:", path, "ABI=", abi())
assert abi() >= 3, "route package predates the registration capability probes"
PY

python3 - <<'PY'
import ctypes
from pathlib import Path
import deep_ep.deep_ep_cpp as ext
path = Path(ext.__file__).resolve()
lib = ctypes.CDLL(str(path))
version = lib.A5DeepEpExplicitMultipathAttrAbiVersion
version.restype = ctypes.c_int
print("DeepEP:", path, "ABI=", version())
assert version() >= 7
for name in ("prepare_ccu_hbm_command_worker", "ccu_hbm_command_puncture",
             "stop_ccu_hbm_command_worker"):
    assert hasattr(ext.Buffer, name), name
PY

run_dir="${output_root}/a5_ccu_hbm_command_puncture_${worker_register_mode}_${src_phy}_${dst_phy}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${run_dir}/plans"
relay_weights=""
for ((i=0; i<${#relay_array[@]}; ++i)); do
    [[ -z "${relay_weights}" ]] || relay_weights+=,
    relay_weights+=1
done
resolve=(python3 "${script_dir}/resolve_a5_explicit_multirelay_eids.py"
    --topology "${topology}" --topology-json "${topology_json}"
    --src-phy "${src_phy}" --dst-phy "${dst_phy}"
    --relay-phys "${relay_phys}" --weights "${relay_weights}")
[[ -z "${relay_planes}" ]] || resolve+=(--relay-planes "${relay_planes}")
"${resolve[@]}" >"${run_dir}/resolved.tsv"
python3 "${script_dir}/prepare_a5_ccu_explicit_multipath_plans.py" \
    --resolved-manifest "${run_dir}/resolved.tsv" --output-dir "${run_dir}/plans" \
    --direct-route "${direct_route}" --direct-relay-ratio 2:1

weights=2
for ((i=0; i<${#relay_array[@]}; ++i)); do weights+=,1; done
args=(--bytes "${bytes}" --relay-manifest "${run_dir}/plans/direct_plus_relays.tsv"
      --direct-route "${direct_route}" --path-weights "${weights}")
(( graph )) && args+=(--graph)
(( register_only )) && args+=(--register-only)

export A5_CCU_WORKER_REGISTER_MODE="${worker_register_mode}"
export A5_CCU_WORKER_REGISTER_ONLY="${register_only}"

status=0
timeout --signal=TERM --kill-after=10 "${timeout_seconds}" \
    python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
    tests/python/deepep/test_a5_ccu_hbm_command_puncture.py "${args[@]}" \
    >"${run_dir}/puncture.log" 2>&1 || status=$?
printf 'status\t%s\n' "${status}" >"${run_dir}/status.tsv"
grep -E 'PUNCTURE_|COMMAND_BLOCK_(WORKER|REGISTER)' "${run_dir}/puncture.log" || true
if (( status != 0 )) || ! grep -q '^PUNCTURE_RESULT PASS' "${run_dir}/puncture.log"; then
    echo "Puncture failed; inspect ${run_dir}/puncture.log" >&2
    tail -120 "${run_dir}/puncture.log" >&2
    (( status != 0 )) || status=1
    exit "${status}"
fi
echo "Result directory: ${run_dir}"
