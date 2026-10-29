#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
devices=2,3
sizes=4096,4194304
copy_mode=both
group_slices_kib=32
changed=3
queued=20
timeout_seconds=300
cann_root=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.3.0}
output_root=/home/l00934901/profiling
builder_only=0
benchmark=0
profile=0
warmup=100
iterations=20
export_profile=1
while (($#)); do
    case "$1" in
        --devices) devices=$2; shift 2;;
        --sizes) sizes=$2; shift 2;;
        --copy-mode) copy_mode=$2; shift 2;;
        --group-slices-kib) group_slices_kib=$2; shift 2;;
        --changed-iters) changed=$2; shift 2;;
        --queued-iters) queued=$2; shift 2;;
        --timeout-seconds) timeout_seconds=$2; shift 2;;
        --cann-root) cann_root=$2; shift 2;;
        --output-root) output_root=$2; shift 2;;
        --builder-only) builder_only=1; shift;;
        --benchmark) benchmark=1; shift;;
        --profile) profile=1; shift;;
        --no-profile) profile=0; shift;;
        --warmup) warmup=$2; shift 2;;
        --iters) iterations=$2; shift 2;;
        --no-export) export_profile=0; shift;;
        --help)
            echo 'Isolated CCU LocalCopyNb + WaitEvent diagnostic; no remote writes or ACL D2D.'
            echo 'Example (on NPU): bash scripts/run_a5_ccu_local_copy_probe.sh --devices 2,3 --output-root /home/l00934901/profiling'
            echo '--devices one/two IDs; --sizes 4096,4194304; --copy-mode direct|chunked|both|group (native-style parallel MS copy)'
            echo '--group-slices-kib 32,64,128 (group only; fixed 64 MS budget: 8x32 / 4x64 / 2x128 KiB)'
            echo '--changed-iters 3 --queued-iters 20 --timeout-seconds 300 --cann-root PATH'
            echo '--builder-only compiles/builds representations without NPU; never counts as hardware PASS.'
            echo '--benchmark --warmup 100 --iters 20 [--profile] [--no-export]; times CCU-only batch after correctness gate.'
            exit 0;;
        *) echo "unsupported argument: $1" >&2; exit 2;;
    esac
done
# Import-safe policy helper only. Reject bad arguments before CANN initialization.
metadata=$(python3 - "$repo/scripts" "$devices" "$sizes" "$copy_mode" "$changed" "$queued" "$timeout_seconds" "$builder_only" "$benchmark" "$profile" "$warmup" "$iterations" "$group_slices_kib" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
from prepare_a5_ccu_peer_plan import parse_phys_ids
try:
    devices = parse_phys_ids(sys.argv[2], '--devices')
    sizes = sorted(parse_phys_ids(sys.argv[3], '--sizes'))
    mode, changed, queued, timeout, builder = sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), bool(int(sys.argv[8]))
    benchmark, profile, warmup, iterations = bool(int(sys.argv[9])), bool(int(sys.argv[10])), int(sys.argv[11]), int(sys.argv[12])
    slices = parse_phys_ids(sys.argv[13], '--group-slices-kib')
    if any(s not in (32,64,128) for s in slices) or (mode != 'group' and slices != [32]):
        raise ValueError('group slice must be 32/64/128 KiB; non-default slices require --copy-mode group')
    if not 0 <= warmup <= 100000 or not 1 <= iterations <= 100000 or (profile and not benchmark):
        raise ValueError('profiling requires --benchmark; warmup 0..100000, iters 1..100000')
    if len(devices) not in (1, 2) or mode not in ('direct', 'chunked', 'both', 'group'):
        raise ValueError('one/two devices and direct|chunked|both|group required')
    if not 1 <= changed <= 1000 or not 1 <= queued <= 32 or timeout < 1:
        raise ValueError('invalid iteration count or timeout')
    if any(n < 256 or n > 64*1024*1024 or n % 256 or
           (mode in ('chunked','both') and n % 4096) or (n+8192)*2*queued > 256*1024*1024 for n in sizes):
        raise ValueError('size must be aligned (chunked:4096, direct/group:256); pinned host budget <=256MiB/rank')
    # Small direct first; chunked controls before the largest single-operation
    # test, so a large-length failure cannot erase the smaller positive evidence.
    pairs = [('direct', sizes[0])] if mode == 'both' else []
    if mode in ('chunked', 'both'): pairs += [('chunked', n) for n in sizes]
    if mode == 'direct': pairs += [('direct', n) for n in sizes]
    elif mode == 'both': pairs += [('direct', n) for n in sizes[1:]]
    cases = [dict(name=f'{m}_{n}', copy_mode=m, bytes=n) for m,n in pairs]
    if mode == 'group':
        cases = [dict(name=f'group_l{256//s}_s{s}k_{n}', copy_mode='group', bytes=n,
            group_slice_bytes=s*1024, group_lanes=256//s, group_ms_budget=64) for s in slices for n in sizes]
    print(json.dumps(dict(devices=devices, changed_calls=changed, queued_calls=queued,
        builder_only=builder, benchmark=benchmark, profile=profile, warmup=warmup, iterations=iterations,
        group_slices_bytes=[s*1024 for s in slices],cases=cases)))
except (ValueError, TypeError) as error:
    print(f'ERROR: {error}', file=sys.stderr); sys.exit(2)
PY
)
[[ -f "${cann_root}/set_env.sh" ]] || { echo "missing CANN set_env.sh: ${cann_root}" >&2; exit 2; }
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX ASCEND_LAUNCH_BLOCKING
set +u
source "${cann_root}/set_env.sh"
set -u
export LD_LIBRARY_PATH="${cann_root}/lib64:${cann_root}/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export ASCEND_RT_VISIBLE_DEVICES="$devices"
mkdir -p "$output_root"
run_dir=$(mktemp -d "${output_root}/a5_ccu_local_copy_${devices//,/_}_$(date +%Y%m%d_%H%M%S).XXXXXX")
mkdir -p "${run_dir}/bin" "${run_dir}/cases"
echo "Result directory: ${run_dir}"
printf '%s\n' "$metadata" > "${run_dir}/run_settings.json"
python3 - "${run_dir}/run_settings.json" > "${run_dir}/cases.tsv" <<'PY'
import json,sys
for case in json.load(open(sys.argv[1]))['cases']:
    print(case['name'],case['copy_mode'],case['bytes'],case.get('group_slice_bytes',32768),sep='\t')
PY
binary="${run_dir}/bin/local_copy_probe"
# Standalone executable: do not install SO/wheel or overwrite the production op.
if ! g++ -std=c++14 -Wall -Wextra -Werror -DHOST_COMPILE  \
    -I"${repo}/examples/a5_ccu_urma_route_probe/testcase" \
    -I"${repo}/examples/a5_ccu_urma_route_probe/common" -I"${cann_root}/include/hcomm/ccu" -I"${cann_root}/include" -I"${cann_root}/include/hccl" -I"${cann_root}/include/hcomm" \
    -I"${cann_root}/pkg_inc" -I"${cann_root}/pkg_inc/hcomm" -I"${cann_root}/pkg_inc/hcomm/ccu" \
    "${repo}/examples/a5_ccu_urma_route_probe/testcase/local_copy_probe.cc" \
    "${repo}/examples/a5_ccu_urma_route_probe/op_kernel_ccu/grouped_local_copy_kernel.cc" \
    "${repo}/examples/a5_ccu_urma_route_probe/op_host/public_ccu.cc" \
    -L"${cann_root}/lib64" -Wl,-rpath,"${cann_root}/lib64" \
    -Wl,--no-as-needed -lhccl -lhcomm -lascendcl -lmsprofiler -Wl,--as-needed -ldl -pthread \
    -o "$binary" > "${run_dir}/build.log" 2>&1; then
    echo "BUILD_FAIL: ${run_dir}/build.log" >&2
    tail -n 60 "${run_dir}/build.log"; exit 1
fi
ranks=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["devices"]))' "${run_dir}/run_settings.json")
if ((builder_only)); then
    for mode in direct chunked; do
        if ! timeout --signal=TERM --kill-after=5 "$timeout_seconds" "$binary" --builder-check --copy-mode "$mode" \
            > "${run_dir}/builder_${mode}.log" 2>&1; then
            cat "${run_dir}/builder_${mode}.log"; exit 1
        fi
        cat "${run_dir}/builder_${mode}.log"
    done
    mapfile -t slices < <(python3 -c 'import json,sys; print(*json.load(open(sys.argv[1]))["group_slices_bytes"],sep="\n")' "${run_dir}/run_settings.json")
    for slice_bytes in "${slices[@]}"; do
        slice=$((slice_bytes/1024))
        if ! timeout --signal=TERM --kill-after=5 "$timeout_seconds" "$binary" --builder-check --copy-mode group \
            --group-slice-bytes "$slice_bytes" > "${run_dir}/builder_group_s${slice}k.log" 2>&1; then
            cat "${run_dir}/builder_group_s${slice}k.log"; exit 1
        fi
        cat "${run_dir}/builder_group_s${slice}k.log"
    done
    echo 'BUILD_ONLY_PASS: representations accepted; no NPU copy executed. This is NOT a hardware PASS.'
    exit 0
fi
export LOCAL_COPY_BINARY="$binary" LOCAL_COPY_RANKS="$ranks" LOCAL_COPY_CHANGED="$changed" LOCAL_COPY_QUEUED="$queued"
export LOCAL_COPY_BENCHMARK="$benchmark" LOCAL_COPY_PROFILE="$profile" LOCAL_COPY_WARMUP="$warmup" LOCAL_COPY_ITERS="$iterations"
while IFS=$'\t' read -r name mode bytes slice_bytes; do
    case_dir="${run_dir}/cases/${name}"; mkdir -p "$case_dir"
    export LOCAL_COPY_CASE_DIR="$case_dir" LOCAL_COPY_MODE="$mode" LOCAL_COPY_BYTES="$bytes"
    export LOCAL_COPY_GROUP_SLICE_BYTES="$slice_bytes"
    set +e
    timeout --signal=TERM --kill-after=5 "$timeout_seconds" bash -c '
        pids=()
        for ((rank=0; rank<LOCAL_COPY_RANKS; rank++)); do
            (
                extra=()
                if ((LOCAL_COPY_BENCHMARK)); then
                    extra+=(--benchmark --warmup "$LOCAL_COPY_WARMUP" --iters "$LOCAL_COPY_ITERS")
                fi
                if ((LOCAL_COPY_PROFILE)); then
                    profile_dir="${LOCAL_COPY_CASE_DIR}/profiling/rank${rank}"
                    mkdir -p "$profile_dir"
                    extra+=(--profile-dir "$profile_dir")
                fi
                "${LOCAL_COPY_BINARY}" --rank "$rank" --ranks "$LOCAL_COPY_RANKS" \
                    --case-dir "$LOCAL_COPY_CASE_DIR" --copy-mode "$LOCAL_COPY_MODE" \
                    --group-slice-bytes "$LOCAL_COPY_GROUP_SLICE_BYTES" \
                    --bytes "$LOCAL_COPY_BYTES" --changed-iters "$LOCAL_COPY_CHANGED" --queued-iters "$LOCAL_COPY_QUEUED" "${extra[@]}" \
                    > "${LOCAL_COPY_CASE_DIR}/rank${rank}.log" 2>&1
                rc=$?; printf "%s\n" "$rc" > "${LOCAL_COPY_CASE_DIR}/rank${rank}.status"; exit "$rc"
            ) &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        exit "$failed"
    '
    rc=$?
    set -e
    printf '%s\n' "$rc" > "${case_dir}/status.txt"
    echo "${name}: process status=${rc}"
    if ((rc)); then
        tail -n 30 "${case_dir}"/rank*.log
        echo 'Stopping after failure; no device reset or subsequent hardware cases.' >&2
        break
    fi
done < "${run_dir}/cases.tsv"
set +e
python3 "${repo}/scripts/analyze_a5_ccu_local_copy_probe.py" --run-dir "$run_dir"
proof_rc=$?
set -e
perf_rc=0
if ((benchmark)); then
    if ((profile && export_profile)); then
        msprof="${cann_root}/tools/profiler/bin/msprof"
        if [[ ! -x "$msprof" ]]; then msprof=$(command -v msprof || true); fi
        if [[ -n "$msprof" && -x "$msprof" ]]; then
            while IFS= read -r profile_dir; do
                set +e
                timeout --signal=TERM --kill-after=5 "$timeout_seconds" "$msprof" \
                    --export=on --output="$profile_dir" > "$profile_dir/export.log" 2>&1
                export_rc=$?
                set -e
                printf '%s\n' "$export_rc" > "$profile_dir/export.status"
                echo "Profiling export: ${profile_dir} status=${export_rc}"
            done < <(find "${run_dir}/cases" -type d -path '*/profiling/rank*' -name 'rank[0-9]*' | sort)
        else
            echo 'MSPROF_EXPORT_UNAVAILABLE: raw PROF data retained; export later on a Toolkit machine.' >&2
        fi
    fi
    set +e
    python3 "${repo}/scripts/analyze_a5_ccu_local_copy_perf.py" --run-dir "$run_dir"
    perf_rc=$?
    set -e
fi
if ((proof_rc || perf_rc)); then exit 1; fi
