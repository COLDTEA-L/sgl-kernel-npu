#!/usr/bin/env bash
# Shared launcher environment; this file defines a function only.
a5_ccu_prepare_eid_log_env() {
    # An explicit launcher option wins; otherwise honor the caller's SDK settings.
    if [[ -n ${A5_EID_LOG_LEVEL:-} ]]; then
        export ASCEND_GLOBAL_LOG_LEVEL=$A5_EID_LOG_LEVEL
        export ASCEND_MODULE_LOG_LEVEL="HCCL=$A5_EID_LOG_LEVEL:HCCP=$A5_EID_LOG_LEVEL"
    else
        export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
        export ASCEND_MODULE_LOG_LEVEL=${ASCEND_MODULE_LOG_LEVEL:-HCCL=$ASCEND_GLOBAL_LOG_LEVEL:HCCP=$ASCEND_GLOBAL_LOG_LEVEL}
    fi
    export ASCEND_SLOG_PRINT_TO_STDOUT=${A5_EID_LOG_STDOUT:-${ASCEND_SLOG_PRINT_TO_STDOUT:-0}}
}

# Only EID create tests opt into a smaller default; preserve explicit settings.
a5_ccu_prepare_eid_buffer_env() {
    local a5_eid_suite=benchmark
    while (($#)); do
        case "$1" in
            --suite) if (($# > 1)); then a5_eid_suite=$2; shift; fi ;;
            --suite=*) a5_eid_suite=${1#--suite=} ;;
        esac
        shift
    done
    if [[ $a5_eid_suite == create && -z ${HCCL_BUFFSIZE:-} ]]; then
        export HCCL_BUFFSIZE=64
    fi
}

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
