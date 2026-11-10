#!/usr/bin/env bash
# EP8 only: end-to-end explicit-EID paths through communicator members.
set -Ee -o pipefail
A5_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$A5_REPO"
source scripts/a5_ccu_test_env.sh
A5_EID_CANN=${A5_CANN_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.3.0}}
a5_ccu_prepare_eid_buffer_env "$@"
a5_ccu_prepare_test_env "$A5_EID_CANN" "$A5_REPO"
export A5_CCU_ROUTE_PROBE_LIB=${A5_CCU_ROUTE_PROBE_LIB:-$A5_EID_CANN/opp/vendors/cust/lib64/liba5_ccu_urma_route_probe.so}
export ASCEND_RT_VISIBLE_DEVICES=${A5_DEVICES:-0,1,2,3,4,5,6,7}
export A5_CCU_CHANNEL_SYNC_MODE=baseline A5_CCU_PG_MODE=env
export PYTHONUNBUFFERED=1
a5_ccu_prepare_eid_log_env
unset HCCL_ALGO
A5_RESULTS_ROOT=${A5_OUTPUT_ROOT:-/home/l00934901/profiling}
mkdir -p "$A5_RESULTS_ROOT"
A5_EID_RUN=$(mktemp -d "$A5_RESULTS_ROOT/a5-ep8-eid.XXXXXX")
mkdir -p "$A5_EID_RUN/tmp"
export TMPDIR="$A5_EID_RUN/tmp"
echo "结果目录：$A5_EID_RUN"
printf 'EID_LOG_ENV global=%s modules=%s stdout=%s resource_diagnostics=%s hccl_buffsize_mb=%s\n' \
  "$ASCEND_GLOBAL_LOG_LEVEL" "$ASCEND_MODULE_LOG_LEVEL" "$ASCEND_SLOG_PRINT_TO_STDOUT" \
  "${A5_CCU_RESOURCE_DIAGNOSTICS:-0}" "$HCCL_BUFFSIZE" | tee "$A5_EID_RUN/environment.log"
printf '%s\n' "$@" > "$A5_EID_RUN/arguments.txt"
git rev-parse HEAD > "$A5_EID_RUN/commit.txt"
python3 scripts/prepare_a5_ccu_ep8_topology.py --executor eid \
  --max-relays-per-peer "${A5_EID_MAX_RELAYS_PER_PEER:-6}" \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" \
  --root-info "${A5_ROOT_INFO:-/etc/hccl_rootinfo.json}" \
  --topology-json "${A5_ATLAS_JSON:-/usr/local/Ascend/driver/topo/950/atlas_950_1.json}" \
  --hccn-tool "${A5_HCCN_TOOL:-/usr/local/Ascend/driver/tools/hccn_tool}" \
  --output-dir "$A5_EID_RUN/plans" > "$A5_EID_RUN/topology.log" 2>&1 || {
  cat "$A5_EID_RUN/topology.log"; exit 1;
}
if timeout --signal=TERM --kill-after=5 "${A5_EID_TIMEOUT:-900}" \
  python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/python/deepep/test_a5_ccu_eid_npu.py \
  --manifest "$A5_EID_RUN/plans/peer_plan.tsv" \
  --topology-json "$A5_EID_RUN/plans/topology.json" \
  --run-dir "$A5_EID_RUN/results" "$@" > "$A5_EID_RUN/run.log" 2>&1; then
  echo 0 > "$A5_EID_RUN/exit.status"
else
  A5_EXIT=$?
  echo "$A5_EXIT" > "$A5_EID_RUN/exit.status"
  tail -100 "$A5_EID_RUN/run.log"
  exit "$A5_EXIT"
fi
python3 scripts/summarize_a5_ccu_eid.py "$A5_EID_RUN"
