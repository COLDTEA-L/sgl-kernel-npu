#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

src_phy=2
dst_phy=3
relay_phys=0,1
relay_planes=""
direct_route=0
bytes=$((4 * 1024 * 1024))
timeout_seconds=180
callback_timeout_ms=60000
direct_weights=1,0,0
scheduled_weights=2,1,1
output_root=/home/l00934901/profiling
topology="${repo_root}/docs/topology/a5_hccn_device_topology_raw.txt"
topology_json=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
hccl_lib_dir=${A5_EXPLICIT_HCCL_LIB_DIR:-}
cann_root=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.T560}

usage() {
    cat <<'EOF'
Usage: run_a5_valuewait_callback_puncture.sh [options]

  --src-phy N              communication source card (default: 2)
  --dst-phy N              communication destination card (default: 3)
  --relay-phys A,B         ordered relay cards (default: 0,1)
  --relay-planes A,B       optional topology plane override
  --direct-route N         HCCL direct candidate index (default: 0)
  --bytes N                bytes per destination slice
  --direct-weights W0,W1,W2      first policy (default: 1,0,0)
  --scheduled-weights W0,W1,W2   second policy (default: 2,1,1)
  --hccl-lib-dir DIR       patched HCCL directory containing libhccl.so
  --cann-root DIR          CANN installation root
  --topology FILE          HCCN text topology
  --topology-json FILE     driver topology JSON
  --output-root DIR        result root
  --timeout-seconds N      whole worker timeout
  --callback-timeout-ms N  per replay host completion timeout
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-phy) src_phy=$2; shift 2 ;;
        --dst-phy) dst_phy=$2; shift 2 ;;
        --relay-phys) relay_phys=$2; shift 2 ;;
        --relay-planes) relay_planes=$2; shift 2 ;;
        --direct-route) direct_route=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --direct-weights) direct_weights=$2; shift 2 ;;
        --scheduled-weights) scheduled_weights=$2; shift 2 ;;
        --hccl-lib-dir) hccl_lib_dir=$2; shift 2 ;;
        --cann-root) cann_root=$2; shift 2 ;;
        --topology) topology=$2; shift 2 ;;
        --topology-json) topology_json=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --timeout-seconds) timeout_seconds=$2; shift 2 ;;
        --callback-timeout-ms) callback_timeout_ms=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value in "$src_phy" "$dst_phy" "$direct_route" "$bytes" \
             "$timeout_seconds" "$callback_timeout_ms"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "invalid numeric argument: $value" >&2; exit 2; }
done
(( src_phy != dst_phy && bytes > 0 && bytes % 256 == 0 &&
   timeout_seconds > 0 && callback_timeout_ms > 0 )) || {
    echo "invalid endpoints/bytes/timeout" >&2; exit 2;
}
IFS=',' read -ra relay_array <<<"${relay_phys}"
(( ${#relay_array[@]} == 2 )) || {
    echo "this puncture requires exactly two relay cards" >&2; exit 2;
}
for relay in "${relay_array[@]}"; do
    [[ "$relay" =~ ^[0-9]+$ ]] || { echo "invalid relay card: $relay" >&2; exit 2; }
    (( relay != src_phy && relay != dst_phy )) || {
        echo "relay card must differ from communication endpoints" >&2; exit 2;
    }
done
(( relay_array[0] != relay_array[1] )) || {
    echo "relay cards must be distinct" >&2; exit 2;
}

IFS=',' read -ra direct_weight_array <<<"${direct_weights}"
IFS=',' read -ra scheduled_weight_array <<<"${scheduled_weights}"
(( ${#direct_weight_array[@]} == 3 && ${#scheduled_weight_array[@]} == 3 )) || {
    echo "this direct+2relay puncture requires three weights per policy" >&2; exit 2;
}
for value in "${direct_weight_array[@]}" "${scheduled_weight_array[@]}"; do
    [[ "$value" =~ ^[0-9]+$ ]] && (( value <= 15 )) || {
        echo "path weights must be integers in [0,15]" >&2; exit 2;
    }
done
(( direct_weight_array[0] > 0 && direct_weight_array[1] == 0 &&
   direct_weight_array[2] == 0 )) || {
    echo "--direct-weights must be direct-only (W0,0,0 with W0>0)" >&2; exit 2;
}
(( scheduled_weight_array[1] > 0 || scheduled_weight_array[2] > 0 )) || {
    echo "--scheduled-weights must enable at least one relay path" >&2; exit 2;
}

[[ -n "${hccl_lib_dir}" ]] || {
    echo "--hccl-lib-dir (or A5_EXPLICIT_HCCL_LIB_DIR) is required" >&2; exit 2;
}
hccl_lib_dir=$(readlink -f "${hccl_lib_dir}")
hccl_so="${hccl_lib_dir}/libhccl.so"
hccl_compat_so="${hccl_lib_dir}/libhccl_compat.so"
[[ -f "${hccl_so}" && -f "${hccl_compat_so}" ]] || {
    echo "missing patched libhccl.so/libhccl_compat.so under ${hccl_lib_dir}" >&2; exit 2;
}

cd "${repo_root}"
source "${repo_root}/scripts/a5_ccu_test_env.sh"
a5_ccu_prepare_test_env "${cann_root}" "${repo_root}"
export ASCEND_RT_VISIBLE_DEVICES="${src_phy},${dst_phy}"
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2300}
export LD_LIBRARY_PATH="${hccl_lib_dir}:${LD_LIBRARY_PATH}"
hccl_preload="${hccl_compat_so}:${hccl_so}"

npu_smi_wrapper_dir="${script_dir}/wrappers"
[[ -x "${npu_smi_wrapper_dir}/npu-smi" ]] || {
    echo "missing npu-smi preload-isolation wrapper: ${npu_smi_wrapper_dir}/npu-smi" >&2
    exit 2
}
worker_path="${npu_smi_wrapper_dir}:${PATH}"

HCCL_COMPAT_SO="${hccl_compat_so}" HCCL_SO="${hccl_so}" \
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" python3 - <<'PY'
import ctypes
import os
ctypes.CDLL(os.environ["HCCL_COMPAT_SO"], mode=ctypes.RTLD_GLOBAL)
hccl = ctypes.CDLL(os.environ["HCCL_SO"], mode=ctypes.RTLD_GLOBAL)
fn = hccl.A5HcclExplicitMultipathExtensionVersion
fn.restype = ctypes.c_int
version = fn()
print(f"Verified patched HCCL explicit multipath extension: {version}")
assert version >= 10, f"patched HCCL extension {version}, expected >= 10"
PY

run_dir="${output_root}/a5_valuewait_callback_${src_phy}_${dst_phy}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${run_dir}/plans"
resolved="${run_dir}/resolved_2_relays.tsv"
resolve=(python3 "${script_dir}/resolve_a5_explicit_multirelay_eids.py"
    --topology "${topology}" --topology-json "${topology_json}"
    --src-phy "${src_phy}" --dst-phy "${dst_phy}"
    --relay-phys "${relay_phys}" --weights 1,1)
[[ -z "${relay_planes}" ]] || resolve+=(--relay-planes "${relay_planes}")
"${resolve[@]}" >"${resolved}"
python3 "${script_dir}/prepare_a5_ccu_explicit_multipath_plans.py" \
    --resolved-manifest "${resolved}" --output-dir "${run_dir}/plans" \
    --direct-route "${direct_route}" --direct-relay-ratio 2:1
manifest="${run_dir}/plans/direct_plus_2relay.tsv"

echo "communication cards: ${src_phy},${dst_phy}"
echo "relay cards:         ${relay_phys}"
echo "policy transition:   ${direct_weights} -> ${scheduled_weights}"
echo "manifest:            ${manifest}"

pg_init_file="${run_dir}/pgstore"
log="${run_dir}/puncture.log"
status=0
timeout --signal=TERM --kill-after=5 "${timeout_seconds}" \
  env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" LD_PRELOAD="${hccl_preload}" \
      PATH="${worker_path}" PYTHONUNBUFFERED=1 \
      A5_CCU_PG_INIT_FILE="${pg_init_file}" \
  python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
    tests/python/deepep/test_a5_valuewait_callback_puncture.py \
    --bytes "${bytes}" --relay-manifest "${manifest}" \
    --direct-weights "${direct_weights}" \
    --scheduled-weights "${scheduled_weights}" \
    --timeout-ms "${callback_timeout_ms}" >"${log}" 2>&1 || status=$?
rm -f "${pg_init_file}"

if (( status != 0 )) || ! grep -q '^PASS:' "${log}"; then
    echo "puncture experiment failed (status=${status}): ${log}" >&2
    tail -n 160 "${log}" >&2
    exit 1
fi
grep -E 'CASE_(WARMUP|GRAPH_CAPTURE|STAGE)|^PASS:' "${log}"
echo "PASS: log=${log}"
