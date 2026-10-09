#!/usr/bin/env bash
# Shared launcher environment; this file defines a function only.
a5_ccu_prepare_test_env() {
    local a5_test_cann=$1 a5_test_repo=$2 a5_test_nounset=0
    [[ $- != *u* ]] || a5_test_nounset=1
    [[ -f "${a5_test_cann}/set_env.sh" ]] || {
        echo "missing CANN set_env.sh: ${a5_test_cann}" >&2; return 1;
    }
    unset LD_PRELOAD
    set +u
    source "${a5_test_cann}/set_env.sh" || {
        ((a5_test_nounset == 0)) || set -u
        return 1
    }
    ((a5_test_nounset == 0)) || set -u
    # Generated vendor set_env.bash may contain the build server's absolute
    # path. Derive this checkout's path; Python then prefers the installed wheel.
    local a5_test_vendor="${a5_test_repo}/python/deep_ep/deep_ep/vendors/hwcomputing"
    if [[ -d "${a5_test_vendor}" ]]; then
        export ASCEND_CUSTOM_OPP_PATH="${a5_test_vendor}:${ASCEND_CUSTOM_OPP_PATH:-}"
        export LD_LIBRARY_PATH="${a5_test_vendor}/op_api/lib:${LD_LIBRARY_PATH:-}"
    fi
    export HCCL_OP_EXPANSION_MODE=CCU_SCHED
    export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2300}
    unset ASCEND_LAUNCH_BLOCKING A5_CCU_DEBUG A5_CCU_PREPARED_ALLOW_LAZY_STREAM
    unset A5_URMA_TP_TRACE_PREFIX
}
