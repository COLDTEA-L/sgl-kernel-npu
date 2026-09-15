#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
probe_dir="${repo_root}/examples/a5_urma_full_eid_probe"
umdk_root=/home/l00934901/umdk
urma_lib_dir=/usr/lib64
output_root=/home/l00934901/profiling
src_phy=6
dst_phy=7

usage() {
  cat <<'EOF'
Usage: run_a5_urma_full_eid_route_validation.sh [options]
  --src-phy N       physical source device (default: 6)
  --dst-phy N       physical destination device (default: 7)
  --umdk-root PATH  UMDK checkout containing URMA headers
  --urma-lib-dir P  directory containing liburma.so
  --output-root P   result root

This gate validates the exact route EIDs observed in HCCL/HCCN. It does not
silently fall back to an ambiguous udmac name plus eid_idx.
EOF
}

while (($#)); do
  case "$1" in
    --src-phy) src_phy=$2; shift 2 ;;
    --dst-phy) dst_phy=$2; shift 2 ;;
    --umdk-root) umdk_root=$2; shift 2 ;;
    --urma-lib-dir) urma_lib_dir=$2; shift 2 ;;
    --output-root) output_root=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${src_phy},${dst_phy}" != "6,7" ]]; then
  echo "No audited full-EID manifest for ${src_phy}->${dst_phy}." >&2
  echo "Generate it from both devices' hccn_tool -g -dev_info output and HCCL route logs first." >&2
  exit 2
fi

make -C "${probe_dir}" UMDK_ROOT="${umdk_root}" URMA_LIB_DIR="${urma_lib_dir}"

timestamp=$(date +%Y%m%d_%H%M%S)
run_dir="${output_root}/a5_urma_full_eid_6_to_7_${timestamp}"
mkdir -p "${run_dir}"
result_tsv="${run_dir}/full_eid_visibility.tsv"
printf 'route\thop\tside\tphysical_device\teid\tstatus\tresolved_device\tresolved_eid_index\tcontext_verified\n' >"${result_tsv}"

# Exact endpoint EIDs copied from the 6/7 HCCN device tables and matched to
# the addresses printed by the HCCL route probe. These are not inferred from
# an EID bit field.
manifest=$(cat <<'EOF'
route0	1	src	6	0000:0000:0046:0300:0010:0000:df12:d706
route0	1	dst	7	0000:0000:0046:0300:0010:0000:df12:f707
route1	2	src	6	0000:0000:003f:0200:0010:0000:df12:cb01
route1	2	dst	7	0000:0000:003f:0200:0010:0000:df12:eb01
route2	2	src	6	0000:0000:007f:0200:0010:0000:df12:db01
route2	2	dst	7	0000:0000:007f:0200:0010:0000:df12:fb01
EOF
)
printf '%b\n' "${manifest}" >"${run_dir}/route_eid_manifest.tsv"

failures=0
while IFS=$'\t' read -r route hop side phy eid; do
  [[ -n "${route}" ]] || continue
  set +e
  output=$(LD_LIBRARY_PATH="${urma_lib_dir}:${LD_LIBRARY_PATH:-}" \
    "${probe_dir}/resolve_full_eid" "${eid}" 2>&1)
  rc=$?
  set -e
  printf '%s\n' "${output}" | tee "${run_dir}/${route}_${side}.log"
  status=$(sed -n 's/.*status=\([^ ]*\).*/\1/p' <<<"${output}" | tail -1)
  dev=$(sed -n 's/.*device=\([^ ]*\).*/\1/p' <<<"${output}" | tail -1)
  idx=$(sed -n 's/.*eid_index=\([^ ]*\).*/\1/p' <<<"${output}" | tail -1)
  verified=$(sed -n 's/.*context_verified=\([^ ]*\).*/\1/p' <<<"${output}" | tail -1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${route}" "${hop}" "${side}" "${phy}" "${eid}" \
    "${status:-UNKNOWN}" "${dev:-N/A}" "${idx:-N/A}" "${verified:-0}" >>"${result_tsv}"
  ((rc == 0)) || failures=$((failures + 1))
done <<<"${manifest}"

echo "Result: ${result_tsv}"
if ((failures)); then
  echo "SCHEME_A_GATE=FAIL: ${failures} route endpoint EID(s) are not exactly bindable by user-space URMA."
  echo "Do not run the old eid_idx pair scan as proof of explicit relay selection."
  exit 1
fi

echo "SCHEME_A_GATE=PASS: all route endpoint EIDs resolved and context binding matched exactly."
echo "Next gate: issue URMA WRITE with the resolved device/eid_index pairs and compare per-port HCCN deltas."
