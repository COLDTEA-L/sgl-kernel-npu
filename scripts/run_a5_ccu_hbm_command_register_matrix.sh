#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runner="${script_dir}/run_a5_ccu_hbm_command_puncture.sh"
output_root=/home/l00934901/profiling
args=("$@")
for ((i=0; i<${#args[@]}; ++i)); do
    if [[ "${args[i]}" == "--output-root" && $((i + 1)) -lt ${#args[@]} ]]; then
        output_root=${args[i + 1]}
    fi
    if [[ "${args[i]}" == "--worker-register-mode" ||
          "${args[i]}" == "--register-only" ]]; then
        echo "do not pass --worker-register-mode/--register-only to the matrix wrapper" >&2
        exit 2
    fi
done

matrix_dir="${output_root}/a5_ccu_hbm_command_register_matrix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${matrix_dir}"
summary="${matrix_dir}/register_matrix.tsv"
printf 'mode\tstatus\tresult\tlog\n' >"${summary}"
printed_abi=0

for mode in hbm_once loop_only loop_hbm full; do
    log="${matrix_dir}/${mode}.log"
    set +e
    bash "${runner}" "${args[@]}" \
        --worker-register-mode "${mode}" --register-only --no-graph \
        >"${log}" 2>&1
    status=$?
    set -e
    if (( printed_abi == 0 )); then
        abi_line=$(grep -m1 '^Route puncture:' "${log}" || true)
        if [[ -n "${abi_line}" ]]; then
            echo "${abi_line}"
            printed_abi=1
        else
            echo "WARNING: route puncture ABI line is missing; inspect ${log}" >&2
        fi
    fi
    run_dir=$(ls -dt "${output_root}/a5_ccu_hbm_command_puncture_${mode}_"* 2>/dev/null | head -1 || true)
    result=FAIL
    if (( status == 0 )) && [[ -n "${run_dir}" ]] &&
       grep -q '^PUNCTURE_RESULT PASS' "${run_dir}/puncture.log"; then
        result=PASS
    fi
    printf '%s\t%s\t%s\t%s\n' "${mode}" "${status}" "${result}" \
        "${run_dir:-${log}}" >>"${summary}"
    echo "${mode}: ${result} (status=${status})"
    if [[ -n "${run_dir}" && -f "${run_dir}/puncture.log" ]]; then
        grep -m1 'COMMAND_BLOCK_REGISTER_TRACE phase=task_arg_ready' \
            "${run_dir}/puncture.log" || true
    fi
done

echo
column -s $'\t' -t "${summary}" 2>/dev/null || cat "${summary}"
echo "Matrix result: ${summary}"
