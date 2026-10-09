#!/usr/bin/env bash
# Native HCCL baseline, not a discovered CommLink or custom direct-only plan.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
devices=0,1,2,3
bytes=4194304
warmup=100
iterations=20
repeats=1
timeout_seconds=600
profile=0
cann_root=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.T560}
output_root=/home/l00934901/profiling
while (($#)); do
    case "$1" in
        --devices) devices=$2; shift 2;;
        --bytes) bytes=$2; shift 2;;
        --warmup) warmup=$2; shift 2;;
        --iters) iterations=$2; shift 2;;
        --repeats) repeats=$2; shift 2;;
        --timeout-seconds) timeout_seconds=$2; shift 2;;
        --profile) profile=1; shift;;
        --cann-root) cann_root=$2; shift 2;;
        --output-root) output_root=$2; shift 2;;
        --help)
            echo 'Native dist.all_to_all_single baseline. --devices 0,1,2,3 --bytes BYTES_PER_PEER --warmup 100 --iters 20 --repeats 3 [--profile]'
            echo 'No DeepEP/route package, relay-map, candidate index or graph capture required.'
            exit 0;;
        *) echo "unsupported argument: $1" >&2; exit 2;;
    esac
done
for value in "$bytes" "$iterations" "$repeats" "$timeout_seconds"; do
    [[ "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]] || { echo 'positive numeric argument required' >&2; exit 2; }
done
[[ "$warmup" =~ ^[0-9]+$ ]] || { echo 'nonnegative warmup required' >&2; exit 2; }
((bytes % 4 == 0)) || { echo '--bytes must be FP32-aligned' >&2; exit 2; }
# Validate before initializing runtime or touching devices.
k=$(python3 - "$devices" <<'PY'
import sys
parts = sys.argv[1].split(',')
if not all(p.isdecimal() for p in parts):
    raise SystemExit('--devices must contain comma-separated nonnegative card IDs')
cards = list(map(int, parts))
if len(cards) not in (2, 4) or len(set(cards)) != len(cards):
    raise SystemExit('--devices must contain two or four distinct physical cards')
print(len(cards))
PY
)
# Match peer-plan environment preparation; never preload our custom runtime.
source "${repo}/scripts/a5_ccu_test_env.sh"
a5_ccu_prepare_test_env "$cann_root" "$repo"
export LD_LIBRARY_PATH="${cann_root}/lib64:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX
unset A5_CCU_ROUTE_INDEX A5_CCU_ROUTE_INDICES A5_CCU_PATH_UIDS A5_CCU_PATH_WEIGHTS
unset A5_CCU_SYNTHETIC_SRC_EID A5_CCU_SYNTHETIC_DST_EID A5_CCU_SYNTHETIC_ROUTE_MANIFEST
export ASCEND_RT_VISIBLE_DEVICES="$devices" PYTHONUNBUFFERED=1
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
mkdir -p "$output_root"
run_dir=$(mktemp -d "${output_root}/a5_ccu_native_alltoall_${devices//,/_}_$(date +%Y%m%d_%H%M%S).XXXXXX")
mkdir -p "${run_dir}/cases"
python3 - "$run_dir" "$devices" "$bytes" "$warmup" "$iterations" "$repeats" "$profile" "$cann_root" <<'PY'
import json, os, pathlib, sys
root, cards, size, warmup, iterations, repeats, profile, cann = sys.argv[1:]
settings = dict(implementation='native_hccl', api='dist.all_to_all_single',
    devices=list(map(int, cards.split(','))), bytes_per_peer=int(size),
    warmup=int(warmup), iterations=int(iterations), repeats=int(repeats),
    profile=bool(int(profile)), cann_root=cann,
    environment={key: os.environ.get(key) for key in
                 ('HCCL_OP_EXPANSION_MODE', 'HCCL_ALGO', 'HCCL_BUFFSIZE', 'LD_LIBRARY_PATH')})
(pathlib.Path(root) / 'run_settings.json').write_text(json.dumps(settings, indent=2) + '\n')
PY
echo "Result directory: ${run_dir}"
profile_args=(); ((profile == 0)) || profile_args=(--profile)
failed=0
for ((r=1; r<=repeats; r++)); do
    case_dir="${run_dir}/r${r}"; mkdir -p "$case_dir"
    log="${run_dir}/cases/native_alltoall_r${r}.log"
    set +e
    timeout --signal=TERM --kill-after=5 "$timeout_seconds" \
        python3 -m torch.distributed.run --standalone --nproc-per-node="$k" \
        "${repo}/tests/python/deepep/test_a5_ccu_native_alltoall.py" \
        --bytes "$bytes" --warmup "$warmup" --iters "$iterations" \
        --run-dir "$case_dir" "${profile_args[@]}" > "$log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" > "${run_dir}/cases/native_alltoall_r${r}.status"
    echo "native_alltoall_r${r}: status=${rc}"
    if ((rc != 0)); then tail -n 80 "$log"; failed=1; break; fi
done
python3 "${repo}/scripts/analyze_a5_ccu_native_alltoall.py" --run-dir "$run_dir"
sed -n '1,200p' "${run_dir}/native_alltoall_report.md"
if ! python3 - "${run_dir}/native_alltoall_summary.json" <<'PY'
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))['complete'] else 1)
PY
then
    failed=1
fi
exit "$failed"
