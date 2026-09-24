#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
inherited_ld_preload=${LD_PRELOAD:-}
unset LD_PRELOAD

src_phy=""
dst_phy=""
relay_phys=""
relay_planes=""
direct_route=0
bytes=4194304
warmup=100
iterations=20
repeats=3
timeout_seconds=300
output_root=/home/l00934901/profiling
topology="${repo_root}/docs/topology/a5_hccn_device_topology_raw.txt"
topology_json=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
profile=0
profile_iters=20
cases=native0,native2,direct_plus_2relay,direct_plus_4relay,direct_plus_6relay
hccl_lib_dir=${A5_EXPLICIT_HCCL_LIB_DIR:-}
cann_root=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.T560}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src-phy) src_phy=$2; shift 2 ;;
        --dst-phy) dst_phy=$2; shift 2 ;;
        --relay-phys) relay_phys=$2; shift 2 ;;
        --relay-planes) relay_planes=$2; shift 2 ;;
        --direct-route) direct_route=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --iters) iterations=$2; shift 2 ;;
        --repeats) repeats=$2; shift 2 ;;
        --timeout-seconds) timeout_seconds=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --topology) topology=$2; shift 2 ;;
        --topology-json) topology_json=$2; shift 2 ;;
        --profile) profile=1; shift ;;
        --profile-iters) profile_iters=$2; shift 2 ;;
        --cases) cases=$2; shift 2 ;;
        --hccl-lib-dir) hccl_lib_dir=$2; shift 2 ;;
        --cann-root) cann_root=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ "${src_phy}" =~ ^[0-9]+$ && "${dst_phy}" =~ ^[0-9]+$ ]] || {
    echo "--src-phy and --dst-phy are required" >&2; exit 2;
}
IFS=',' read -ra relay_array <<<"${relay_phys}"
(( ${#relay_array[@]} == 6 )) || {
    echo "--relay-phys must explicitly list exactly six ordered relay cards" >&2; exit 2;
}
[[ "${direct_route}" =~ ^[0-9]+$ ]] || { echo "invalid --direct-route" >&2; exit 2; }
(( bytes > 0 && bytes % 256 == 0 && warmup >= 0 && iterations > 0 &&
   repeats > 0 && profile_iters > 0 )) || {
    echo "invalid bytes/warmup/iters/repeats/profile-iters" >&2; exit 2;
}
IFS=',' read -ra case_array <<<"${cases}"
(( ${#case_array[@]} > 0 )) || { echo "--cases must not be empty" >&2; exit 2; }
for selected_case in "${case_array[@]}"; do
    case "${selected_case}" in
        native0|native2|direct_plus_2relay|direct_plus_4relay|direct_plus_6relay) ;;
        *) echo "unsupported --cases entry: ${selected_case}" >&2; exit 2 ;;
    esac
done
case_selected() {
    local wanted=$1 item
    for item in "${case_array[@]}"; do
        [[ "${item}" == "${wanted}" ]] && return 0
    done
    return 1
}

cd "${repo_root}"
source "${cann_root}/set_env.sh"
set +u
source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash
set -u
export ASCEND_RT_VISIBLE_DEVICES="${src_phy},${dst_phy}"
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2300}
unset A5_CCU_DEBUG ASCEND_LAUNCH_BLOCKING
if [[ -z "${hccl_lib_dir}" || ! -f "${hccl_lib_dir}/libhccl.so" ||
      ! -f "${hccl_lib_dir}/libhccl_compat.so" ]]; then
    echo "--hccl-lib-dir must name the packaged lib64 directory containing matching patched libhccl.so and libhccl_compat.so" >&2
    exit 2
fi
hccl_so=$(readlink -f "${hccl_lib_dir}/libhccl.so")
hccl_compat_so=$(readlink -f "${hccl_lib_dir}/libhccl_compat.so")
export LD_LIBRARY_PATH="${hccl_lib_dir}:${LD_LIBRARY_PATH:-}"
marker=$(nm -D "${hccl_so}" 2>/dev/null |
    grep -c ' A5HcclExplicitMultipathExtensionVersion$' || true)
(( marker == 1 )) || {
    echo "patched libhccl.so is missing A5HcclExplicitMultipathExtensionVersion" >&2
    exit 2
}
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" python3 -S - "${hccl_so}" <<'PY'
import ctypes
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve()
lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
version = lib.A5HcclExplicitMultipathExtensionVersion
version.restype = ctypes.c_int
if version() < 1:
    raise SystemExit("invalid explicit multipath HCCL extension version")
print(f"Verified patched HCCL: {path} (extension={version()})")
PY
hccl_preload="${hccl_compat_so}:${hccl_so}"
if [[ -n "${inherited_ld_preload}" ]]; then
    echo "Ignoring inherited LD_PRELOAD while running the matrix: ${inherited_ld_preload}" >&2
fi
if [[ -n "${A5_EXPLICIT_EXTRA_LD_PRELOAD:-}" ]]; then
    hccl_preload="${hccl_preload}:${A5_EXPLICIT_EXTRA_LD_PRELOAD}"
fi

if case_selected native0 || case_selected native2; then
    route_probe_lib=${A5_CCU_ROUTE_PROBE_LIB:-}
    if [[ -z "${route_probe_lib}" ]]; then
        for candidate in \
            "${cann_root}/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so" \
            /usr/local/Ascend/cann-9.1.T560/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so
        do
            if [[ -f "${candidate}" ]]; then
                route_probe_lib=${candidate}
                break
            fi
        done
    fi
    [[ -f "${route_probe_lib}" ]] || {
        echo "native0/native2 require liba5_ccu_urma_route_probe.so; install the latest route probe or omit them with --cases" >&2
        exit 2
    }
    nm -D "${route_probe_lib}" | grep -q ' HcclCcuUrmaMultiRouteAllToAll$' || {
        echo "route probe is stale: HcclCcuUrmaMultiRouteAllToAll is missing from ${route_probe_lib}" >&2
        exit 2
    }
    route_probe_lib=$(readlink -f "${route_probe_lib}")
    export A5_CCU_ROUTE_PROBE_LIB="${route_probe_lib}"
    export LD_LIBRARY_PATH="$(dirname "${route_probe_lib}"):${LD_LIBRARY_PATH}"
    echo "Verified route probe: ${route_probe_lib}"
fi

env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" LD_PRELOAD="${hccl_preload}" \
python3 - <<'PY'
import ctypes
from pathlib import Path
import deep_ep.deep_ep_cpp as ext

extension_path = Path(ext.__file__).resolve()
print(f"Verified DeepEP extension: {extension_path}")
assert hasattr(ext.Buffer, "explicit_multipath_all2all_ccu")
assert hasattr(ext.Buffer, "ccu_urma_multiroute_alltoall_out")
try:
    extension = ctypes.CDLL(str(extension_path))
    abi_version = extension.A5DeepEpExplicitMultipathAttrAbiVersion
except AttributeError as error:
    raise RuntimeError(
        "loaded deep_ep_cpp is stale: explicit-multipath ACLNN ABI marker is missing; "
        "rebuild and force-reinstall the wheel from the current branch"
    ) from error
abi_version.restype = ctypes.c_int
version = abi_version()
if version < 2:
    raise RuntimeError(
        f"loaded deep_ep_cpp has explicit-multipath ACLNN ABI {version}, expected >= 2; "
        "rebuild and force-reinstall the wheel from the current branch"
    )
print(f"Verified matrix APIs: explicit + legacy multiroute; attr ABI={version}")
PY

run_dir="${output_root}/a5_ccu_explicit_multipath_${src_phy}_${dst_phy}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${run_dir}/cases" "${run_dir}/plans" "${run_dir}/profiling"
resolved="${run_dir}/resolved_six_relays.tsv"
resolve=(python3 "${script_dir}/resolve_a5_explicit_multirelay_eids.py"
    --topology "${topology}" --topology-json "${topology_json}"
    --src-phy "${src_phy}" --dst-phy "${dst_phy}"
    --relay-phys "${relay_phys}" --weights 1,1,1,1,1,1)
[[ -z "${relay_planes}" ]] || resolve+=(--relay-planes "${relay_planes}")
"${resolve[@]}" >"${resolved}"

python3 "${script_dir}/prepare_a5_ccu_explicit_multipath_plans.py" \
    --resolved-manifest "${resolved}" --output-dir "${run_dir}/plans" \
    --direct-route "${direct_route}" --direct-relay-ratio 2:1

printf 'case\trepeat\tstatus\tresult\n' >"${run_dir}/case_status.tsv"
test_script=tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py
profile_args=()
if (( profile )); then
    profile_args=(--profile --profile-iters "${profile_iters}"
                  --profile-root "${run_dir}/profiling")
fi

run_case() {
    local name=$1 round=$2
    shift 2
    local log="${run_dir}/cases/${name}_r${round}.log" status=0 result=FAIL
    local pg_init_file="${run_dir}/cases/${name}_r${round}.pgstore"
    rm -f "${pg_init_file}"
    timeout --signal=TERM --kill-after=5 "${timeout_seconds}" \
      env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" LD_PRELOAD="${hccl_preload}" \
        PYTHONUNBUFFERED=1 A5_CCU_PHASE_WATCHDOG_SECONDS=60 \
        A5_CCU_PG_INIT_FILE="${pg_init_file}" \
      python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
        "${test_script}" --bytes "${bytes}" --warmup "${warmup}" \
        --iters "${iterations}" "${profile_args[@]}" "$@" \
        >"${log}" 2>&1 || status=$?
    rm -f "${pg_init_file}"
    grep -q '^PASS:' "${log}" && result=PASS
    printf '%s\t%s\t%s\t%s\n' "${name}" "${round}" "${status}" "${result}" \
        >>"${run_dir}/case_status.tsv"
    echo "${name}_r${round}: ${result} (status=${status})"
    if [[ "${result}" != PASS || "${status}" != 0 ]]; then
        echo "===== ${name}_r${round} first errors =====" >&2
        grep -nEi \
          'CASE_FAILURE|traceback|runtimeerror|importerror|attributeerror|undefined symbol|dlopen|not found|failed|error' \
          "${log}" | head -80 >&2 || true
        echo "===== ${name}_r${round} phase trace =====" >&2
        grep -nE 'CASE_PHASE|Timeout \(|Current thread|Thread 0x|File "' \
          "${log}" | tail -120 >&2 || true
        echo "===== ${name}_r${round} log tail =====" >&2
        tail -n 80 "${log}" >&2 || true
    fi
}

for ((round=1; round<=repeats; ++round)); do
    case_selected native0 && run_case native0 "${round}" --implementation multiroute \
        --route-index 0 --path-weights 1 --schedule concurrent
    case_selected native2 && run_case native2 "${round}" --implementation multiroute \
        --route-index 2 --path-weights 1 --schedule concurrent
    case_selected direct_plus_2relay && \
      run_case direct_plus_2relay "${round}" --implementation standard \
        --plan-id "explicit-2relay" --direct-route "${direct_route}" \
        --relay-manifest "${run_dir}/plans/direct_plus_2relay.tsv" \
        --path-weights 4,1,1 --schedule concurrent
    case_selected direct_plus_4relay && \
      run_case direct_plus_4relay "${round}" --implementation standard \
        --plan-id "explicit-4relay" --direct-route "${direct_route}" \
        --relay-manifest "${run_dir}/plans/direct_plus_4relay.tsv" \
        --path-weights 8,1,1,1,1 --schedule concurrent
    case_selected direct_plus_6relay && \
      run_case direct_plus_6relay "${round}" --implementation standard \
        --plan-id "explicit-6relay" --direct-route "${direct_route}" \
        --relay-manifest "${run_dir}/plans/direct_plus_6relay.tsv" \
        --path-weights 12,1,1,1,1,1,1 --schedule concurrent
done

python3 "${script_dir}/analyze_a5_ccu_explicit_multipath_perf.py" --run-dir "${run_dir}"
echo "Result directory: ${run_dir}"
column -s $'\t' -t "${run_dir}/case_status.tsv" 2>/dev/null || cat "${run_dir}/case_status.tsv"
sed -n '1,220p' "${run_dir}/explicit_multipath_perf_report.md"
