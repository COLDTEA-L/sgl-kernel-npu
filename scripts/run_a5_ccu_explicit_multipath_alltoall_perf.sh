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
graph_backend=none
replay_path_weights=""
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
        --graph-backend) graph_backend=$2; shift 2 ;;
        --replay-path-weights) replay_path_weights=$2; shift 2 ;;
        --cases) cases=$2; shift 2 ;;
        --hccl-lib-dir) hccl_lib_dir=$2; shift 2 ;;
        --cann-root) cann_root=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "${graph_backend}" in
    none|eager|aot_eager|npu|npugraphs|npugraph_ex|inductor|aclgraph) ;;
    *) echo "unsupported --graph-backend: ${graph_backend}" >&2; exit 2 ;;
esac

[[ "${src_phy}" =~ ^[0-9]+$ && "${dst_phy}" =~ ^[0-9]+$ ]] || {
    echo "--src-phy and --dst-phy are required" >&2; exit 2;
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
        native0|native2|direct_plus_relays|direct_plus_2relay|direct_plus_4relay|direct_plus_6relay) ;;
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

fixed_relay_count=0
required_relay_count=0
need_legacy=0
need_standard=0
generic_relay_case=0
if case_selected native0 || case_selected native2; then
    need_legacy=1
fi
if case_selected direct_plus_2relay; then
    fixed_relay_count=2
    need_standard=1
fi
if case_selected direct_plus_4relay; then
    fixed_relay_count=4
    need_standard=1
fi
if case_selected direct_plus_6relay; then
    fixed_relay_count=6
    need_standard=1
fi
if case_selected direct_plus_relays; then
    generic_relay_case=1
    need_standard=1
fi

relay_array=()
if (( need_standard )); then
    case "${graph_backend}" in
        none|aclgraph) ;;
        *)
            echo "standard dynamic-policy cases support --graph-backend none or aclgraph" >&2
            exit 2
            ;;
    esac
    IFS=',' read -ra relay_array <<<"${relay_phys}"
    (( ${#relay_array[@]} >= 1 && ${#relay_array[@]} <= 12 )) || {
        echo "standard cases require 1..12 ordered relay cards in --relay-phys" >&2
        exit 2
    }
    required_relay_count=${#relay_array[@]}
    if (( generic_relay_case == 0 && required_relay_count != fixed_relay_count )); then
        echo "selected benchmark cases require exactly ${fixed_relay_count} ordered relay cards; " \
             "use --cases direct_plus_relays to consume an arbitrary relay list" >&2
        exit 2
    fi
    if (( generic_relay_case != 0 && required_relay_count < fixed_relay_count )); then
        echo "the relay list is too short for the selected fixed benchmark cases" >&2
        exit 2
    fi
elif [[ -n "${relay_phys}" ]]; then
    IFS=',' read -ra relay_array <<<"${relay_phys}"
fi

cd "${repo_root}"
source "${cann_root}/set_env.sh"
set +u
source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash
set -u
# Save the unmodified tool environment before prepending the experimental
# HCCL runtime.  Triton's Ascend backend spawns npu-smi while torch_npu is
# imported, and npu-smi must not inherit the worker's private libhccl preload.
a5_real_npu_smi=$(command -v npu-smi || true)
a5_npu_smi_ld_library_path=${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES="${src_phy},${dst_phy}"
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2300}
unset A5_CCU_DEBUG ASCEND_LAUNCH_BLOCKING
if [[ -n "${inherited_ld_preload}" ]]; then
    echo "Ignoring inherited LD_PRELOAD: ${inherited_ld_preload}" >&2
fi
hccl_preload=""
if (( need_standard )); then
    [[ -n "${hccl_lib_dir}" ]] || {
        echo "standard cases require --hccl-lib-dir pointing at the patched HCCL runtime" >&2
        exit 2
    }
    hccl_lib_dir=$(readlink -f "${hccl_lib_dir}")
    hccl_so="${hccl_lib_dir}/libhccl.so"
    hccl_compat_so="${hccl_lib_dir}/libhccl_compat.so"
    [[ -f "${hccl_so}" && -f "${hccl_compat_so}" ]] || {
        echo "missing libhccl.so or libhccl_compat.so under ${hccl_lib_dir}" >&2
        exit 2
    }
    HCCL_COMPAT_SO="${hccl_compat_so}" HCCL_SO="${hccl_so}" \
      LD_LIBRARY_PATH="${hccl_lib_dir}:${LD_LIBRARY_PATH}" python3 - <<'PY'
import ctypes
import os

ctypes.CDLL(os.environ["HCCL_COMPAT_SO"], mode=ctypes.RTLD_GLOBAL)
hccl = ctypes.CDLL(os.environ["HCCL_SO"], mode=ctypes.RTLD_GLOBAL)
version = hccl.A5HcclExplicitMultipathExtensionVersion
version.restype = ctypes.c_int
value = version()
print(f"Verified patched HCCL: {os.environ['HCCL_SO']} (extension={value})")
assert value >= 4, f"patched HCCL extension {value}, expected >= 4 (ALLTOALL + CCU_SCHED entry)"
PY
    export LD_LIBRARY_PATH="${hccl_lib_dir}:${LD_LIBRARY_PATH}"
    hccl_preload="${hccl_compat_so}:${hccl_so}"
fi

worker_path=${PATH}
if [[ -n "${hccl_preload}" ]]; then
    npu_smi_wrapper_dir="${script_dir}/wrappers"
    [[ -x "${npu_smi_wrapper_dir}/npu-smi" ]] || {
        echo "missing executable npu-smi preload-isolation wrapper: ${npu_smi_wrapper_dir}/npu-smi" >&2
        exit 2
    }
    worker_path="${npu_smi_wrapper_dir}:${PATH}"
fi

if (( need_legacy )); then
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
        echo "the selected cases require liba5_ccu_urma_route_probe.so; install the latest route probe" >&2
        exit 2
    }
    if (( need_legacy )); then
        nm -D "${route_probe_lib}" | grep ' HcclCcuUrmaMultiRouteAllToAll$' >/dev/null || {
            echo "route probe is stale: legacy discovered-path API is missing from ${route_probe_lib}" >&2
            exit 2
        }
    fi
    route_probe_lib=$(readlink -f "${route_probe_lib}")
    export A5_CCU_ROUTE_PROBE_LIB="${route_probe_lib}"
    export LD_LIBRARY_PATH="$(dirname "${route_probe_lib}"):${LD_LIBRARY_PATH}"
    echo "Verified route probe: ${route_probe_lib}"
fi

env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    A5_CCU_REQUIRE_LEGACY="${need_legacy}" \
    A5_CCU_REQUIRE_STANDARD="${need_standard}" \
python3 - <<'PY'
import ctypes
import os
from pathlib import Path
import torch
import deep_ep.deep_ep_cpp as ext

extension_path = Path(ext.__file__).resolve()
print(f"Verified DeepEP extension: {extension_path}")
need_legacy = os.environ["A5_CCU_REQUIRE_LEGACY"] == "1"
need_standard = os.environ["A5_CCU_REQUIRE_STANDARD"] == "1"
if need_legacy:
    assert hasattr(ext.Buffer, "ccu_urma_multiroute_alltoall_out")
if need_standard:
    assert hasattr(ext.Buffer, "explicit_multipath_all2all_ccu")
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
if version < 8:
    raise RuntimeError(
        f"loaded deep_ep_cpp has explicit-multipath ABI {version}, expected >= 8; "
        "rebuild and force-reinstall the wheel from the current branch"
    )
print(f"Verified requested APIs: legacy={need_legacy} standard={need_standard}; "
      f"DeepEP ABI={version}")
PY

run_dir="${output_root}/a5_ccu_explicit_multipath_${src_phy}_${dst_phy}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${run_dir}/cases" "${run_dir}/plans" "${run_dir}/profiling"
if (( need_standard )); then
    resolved="${run_dir}/resolved_${required_relay_count}_relays.tsv"
    relay_weights=""
    for ((i=0; i<required_relay_count; ++i)); do
        [[ -z "${relay_weights}" ]] || relay_weights+=,
        relay_weights+=1
    done
    resolve=(python3 "${script_dir}/resolve_a5_explicit_multirelay_eids.py"
        --topology "${topology}" --topology-json "${topology_json}"
        --src-phy "${src_phy}" --dst-phy "${dst_phy}"
        --relay-phys "${relay_phys}" --weights "${relay_weights}")
    [[ -z "${relay_planes}" ]] || resolve+=(--relay-planes "${relay_planes}")
    "${resolve[@]}" >"${resolved}"

    python3 "${script_dir}/prepare_a5_ccu_explicit_multipath_plans.py" \
        --resolved-manifest "${resolved}" --output-dir "${run_dir}/plans" \
        --direct-route "${direct_route}" --direct-relay-ratio 2:1

    generic_path_weights=2
    for ((i=0; i<required_relay_count; ++i)); do
        generic_path_weights+=,1
    done
fi

printf 'case\trepeat\tstatus\tresult\n' >"${run_dir}/case_status.tsv"
test_script=tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py
profile_args=()
if (( profile )); then
    profile_args=(--profile --profile-iters "${profile_iters}"
                  --profile-root "${run_dir}/profiling")
fi
replay_args=()
if [[ -n "${replay_path_weights}" ]]; then
    replay_args=(--replay-path-weights "${replay_path_weights}")
fi

run_case() {
    local name=$1 round=$2
    shift 2
    local log="${run_dir}/cases/${name}_r${round}.log" status=0 result=FAIL
    local pg_init_file="${run_dir}/cases/${name}_r${round}.pgstore"
    rm -f "${pg_init_file}"
    timeout --signal=TERM --kill-after=5 "${timeout_seconds}" \
      env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" LD_PRELOAD="${hccl_preload}" \
        PATH="${worker_path}" A5_REAL_NPU_SMI="${a5_real_npu_smi}" \
        A5_NPU_SMI_LD_LIBRARY_PATH="${a5_npu_smi_ld_library_path}" \
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
    case_selected direct_plus_relays && \
      run_case direct_plus_relays "${round}" --implementation standard \
        --compile-backend "${graph_backend}" \
        --plan-id "explicit-${required_relay_count}relay" --direct-route "${direct_route}" \
        --relay-manifest "${run_dir}/plans/direct_plus_relays.tsv" \
        --path-weights "${generic_path_weights}" "${replay_args[@]}" \
        --schedule concurrent
    case_selected direct_plus_2relay && \
      run_case direct_plus_2relay "${round}" --implementation standard \
        --compile-backend "${graph_backend}" \
        --plan-id "explicit-2relay" --direct-route "${direct_route}" \
        --relay-manifest "${run_dir}/plans/direct_plus_2relay.tsv" \
        --path-weights 2,1,1 --schedule concurrent
    case_selected direct_plus_4relay && \
      run_case direct_plus_4relay "${round}" --implementation standard \
        --compile-backend "${graph_backend}" \
        --plan-id "explicit-4relay" --direct-route "${direct_route}" \
        --relay-manifest "${run_dir}/plans/direct_plus_4relay.tsv" \
        --path-weights 2,1,1,1,1 --schedule concurrent
    case_selected direct_plus_6relay && \
      run_case direct_plus_6relay "${round}" --implementation standard \
        --compile-backend "${graph_backend}" \
        --plan-id "explicit-6relay" --direct-route "${direct_route}" \
        --relay-manifest "${run_dir}/plans/direct_plus_6relay.tsv" \
        --path-weights 2,1,1,1,1,1,1 --schedule concurrent
done

python3 "${script_dir}/analyze_a5_ccu_explicit_multipath_perf.py" --run-dir "${run_dir}"
echo "Result directory: ${run_dir}"
column -s $'\t' -t "${run_dir}/case_status.tsv" 2>/dev/null || cat "${run_dir}/case_status.tsv"
sed -n '1,220p' "${run_dir}/explicit_multipath_perf_report.md"
