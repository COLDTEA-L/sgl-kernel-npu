#!/usr/bin/env bash
set -Eeuo pipefail
A5_ROOT=$(cd "$(dirname "$0")/.." && pwd)
A5_CANN_PATH=${A5_CANN_PATH:-/usr/local/Ascend/cann-9.3.0}
set +u
source "$A5_CANN_PATH/set_env.sh"
set -u
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-64}
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=${ASCEND_SLOG_PRINT_TO_STDOUT:-0}
export ASCEND_RT_VISIBLE_DEVICES=${A5_DEVICES:-0,1}
export LD_LIBRARY_PATH="$A5_CANN_PATH/lib64:${LD_LIBRARY_PATH:-}"
exec python3 "$A5_ROOT/scripts/a5_ccu_l0_l1_probe.py" "$@"
