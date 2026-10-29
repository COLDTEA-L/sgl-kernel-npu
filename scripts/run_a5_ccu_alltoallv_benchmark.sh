#!/usr/bin/env bash
# Eight idle cards required. Writes logs/JSON only; does not enable a profiler.
set -Ee -o pipefail
A5_BENCH_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$A5_BENCH_REPO"
source scripts/a5_ccu_test_env.sh
a5_ccu_prepare_test_env "${A5_CANN_PATH:-/usr/local/Ascend/cann-9.3.0}" "$A5_BENCH_REPO"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export A5_CCU_CHANNEL_SYNC_MODE=baseline
export A5_CCU_PG_MODE=env
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_MODULE_LOG_LEVEL=HCCL=3:HCCP=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
unset HCCL_ALGO
A5_BENCH_ROOT=${A5_OUTPUT_ROOT:-/home/l00934901/profiling}
mkdir -p "$A5_BENCH_ROOT"
A5_BENCH_RUN=$(mktemp -d "$A5_BENCH_ROOT/a5-ep8-benchmark.XXXXXX")
# Direct all temporary/runtime files owned by this launcher to the result directory.
mkdir -p "$A5_BENCH_RUN/tmp"
export TMPDIR="$A5_BENCH_RUN/tmp"
echo "八卡计时结果目录：$A5_BENCH_RUN"
python3 scripts/prepare_a5_ccu_peer_plan.py \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" --available-phys 0,1,2,3,4,5,6,7 \
  --direct-route 0 --output-dir "$A5_BENCH_RUN/plans"
if timeout --signal=TERM --kill-after=5 600 \
  python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/python/deepep/test_a5_ccu_alltoallv_npu.py \
  --manifest "$A5_BENCH_RUN/plans/peer_plan.tsv" \
  --run-dir "$A5_BENCH_RUN/results" --benchmark \
  --dtype float16 --width 4096 --row-scale 128 --warmup 5 --iters 20 --rounds 3 \
  "$@" > "$A5_BENCH_RUN/run.log" 2>&1; then
  echo 0 > "$A5_BENCH_RUN/exit.status"
else
  A5_BENCH_STATUS=$?
  echo "$A5_BENCH_STATUS" > "$A5_BENCH_RUN/exit.status"
  tail -80 "$A5_BENCH_RUN/run.log"
  exit "$A5_BENCH_STATUS"
fi
python3 - "$A5_BENCH_RUN" <<'PY'
import json
import statistics
import sys
from pathlib import Path
root = Path(sys.argv[1])
results = [json.loads((root / 'results' / f'rank{r}.json').read_text()) for r in range(8)]
for rank, result in enumerate(results):
    assert result['rank'] == rank and result['ranks'] == 8
    assert result['correctness'] == 'PASS' and result['cases'] == 1
    assert result['native_reference'] is True and 'timings' in result
summary = {'correctness': 'PASS', 'ranks': 8, 'dtype': results[0]['dtype'],
           'width': results[0]['width'], 'row_scale': results[0]['row_scale'],
           'timing_scope': results[0]['timing_scope'],
           'network_bytes_per_call': results[0]['traffic']['network_bytes'],
           'max_flow_bytes': results[0]['traffic']['max_flow_bytes']}
for name in ('custom', 'native'):
    slowest = [max(r['timings'][name]['round_us'][i] for r in results)
               for i in range(results[0]['rounds'])]
    summary[name] = {'slowest_rank_round_us': slowest,
                     'mean_us': statistics.mean(slowest)}
summary['native_over_custom'] = summary['native']['mean_us'] / summary['custom']['mean_us']
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('EP8_BENCHMARK_PASS')
print(json.dumps(summary, indent=2))
PY
