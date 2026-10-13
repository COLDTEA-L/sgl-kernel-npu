#!/usr/bin/env bash
# Two-card pinned-host H2D/D2H contention sweep; no HCCL/DeepEP/route package.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
devices=2,3
sizes=65536,262144,1048576,4194304,16777216
cases=h2d_solo_rank0,h2d_solo_rank1,h2d_both,d2h_solo_rank0,d2h_solo_rank1,d2h_both,mixed_rank0_h2d_rank1_d2h,mixed_rank0_d2h_rank1_h2d
warmup=5
iterations=20
repeats=3
timeout_seconds=600
cann_root=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.T560}
output_root=/home/l00934901/profiling
while (($#)); do
    case "$1" in
        --devices) devices=$2; shift 2;;
        --sizes) sizes=$2; shift 2;;
        --cases) cases=$2; shift 2;;
        --warmup) warmup=$2; shift 2;;
        --iters) iterations=$2; shift 2;;
        --repeats) repeats=$2; shift 2;;
        --timeout-seconds) timeout_seconds=$2; shift 2;;
        --cann-root) cann_root=$2; shift 2;;
        --output-root) output_root=$2; shift 2;;
        --help)
            echo 'Pinned-host H2D/D2H contention test. Defaults: --devices 2,3 --sizes 65536,262144,1048576,4194304,16777216 --warmup 5 --iters 20 --repeats 3.'
            echo 'Runs solo, same-direction concurrent, and mixed-direction controls in one two-worker process lifetime.'
            echo 'No HCCL communicator, DeepEP extension, custom route library, relay card, profiler, or system route change is used.'
            exit 0;;
        *) echo "unsupported argument: $1" >&2; exit 2;;
    esac
done
for value in "$iterations" "$repeats" "$timeout_seconds"; do
    [[ "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]] || { echo 'positive numeric argument required' >&2; exit 2; }
done
[[ "$warmup" =~ ^[0-9]+$ ]] || { echo 'nonnegative --warmup required' >&2; exit 2; }
python3 - "$devices" "$sizes" <<'PY'
import sys
cards = sys.argv[1].split(',')
if len(cards) != 2 or len(set(cards)) != 2 or any(not item.isdecimal() for item in cards):
    raise SystemExit('--devices must contain exactly two distinct nonnegative physical card IDs')
sizes = sys.argv[2].split(',')
if not sizes or len(sizes) != len(set(sizes)) or any(not item.isdecimal() or int(item) <= 0 for item in sizes):
    raise SystemExit('--sizes must contain unique positive byte counts')
PY
python3 "${repo}/tests/python/deepep/test_a5_h2d_io_contention.py" --help >/dev/null
source "${repo}/scripts/a5_ccu_test_env.sh"
a5_ccu_prepare_test_env "$cann_root" "$repo"
export LD_LIBRARY_PATH="${cann_root}/lib64:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD A5_CCU_ROUTE_PROBE_LIB A5_URMA_TP_TRACE_PREFIX
unset HCCL_OP_EXPANSION_MODE HCCL_ALGO HCCL_BUFFSIZE
export ASCEND_RT_VISIBLE_DEVICES="$devices" PYTHONUNBUFFERED=1
mkdir -p "$output_root"
run_dir=$(mktemp -d "${output_root}/a5_h2d_io_${devices//,/_}_$(date +%Y%m%d_%H%M%S).XXXXXX")
mkdir -p "${run_dir}/cases"
python3 - "$run_dir" "$devices" "$sizes" "$cases" "$warmup" "$iterations" "$repeats" "$cann_root" <<'PY'
import json, pathlib, sys
root, devices, sizes, cases, warmup, iterations, repeats, cann = sys.argv[1:]
settings = dict(
    implementation='torch_npu_pinned_host_copy',
    devices=list(map(int, devices.split(','))),
    sizes=list(map(int, sizes.split(','))),
    cases=cases.split(','),
    warmup=int(warmup), iterations=int(iterations), repeats=int(repeats),
    cann_root=cann, communicator='none', dtype='uint8')
(pathlib.Path(root) / 'run_settings.json').write_text(json.dumps(settings, indent=2) + '\n')
PY
echo "Result directory: ${run_dir}"
log="${run_dir}/cases/h2d_io.log"
set +e
timeout --signal=TERM --kill-after=5 "$timeout_seconds" \
    python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
    "${repo}/tests/python/deepep/test_a5_h2d_io_contention.py" \
    --sizes "$sizes" --cases "$cases" --warmup "$warmup" --iters "$iterations" \
    --repeats "$repeats" --run-dir "$run_dir" >"$log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" >"${run_dir}/cases/h2d_io.status"
if ((rc != 0)); then
    echo "H2D/D2H worker failed: status=${rc}"
    tail -n 100 "$log"
    exit "$rc"
fi
python3 "${repo}/scripts/analyze_a5_h2d_io_contention.py" --run-dir "$run_dir"
sed -n '1,240p' "${run_dir}/h2d_io_report.md"
