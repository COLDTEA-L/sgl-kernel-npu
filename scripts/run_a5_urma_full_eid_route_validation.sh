#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
probe_dir="${repo_root}/examples/a5_urma_full_eid_probe"
umdk_root=/home/l00934901/umdk
urma_include=""
urma_lib_dir=/lib64
output_root=/home/l00934901/profiling
src_phy=6
dst_phy=7
route_log=""
route_discovery_timeout=60

usage() {
  cat <<'EOF'
Usage: run_a5_urma_full_eid_route_validation.sh [options]
  --src-phy N       physical source device (default: 6)
  --dst-phy N       physical destination device (default: 7)
  --umdk-root PATH  UMDK checkout containing URMA headers
  --urma-include P  directory containing urma_api.h (normally auto-detected)
  --urma-lib-dir P  directory containing liburma.so
  --output-root P   result root
  --route-log FILE  parse an existing HCCL route-probe log instead of running it
  --route-discovery-timeout N  seconds allowed for automatic route inventory

This gate validates the exact route EIDs observed in HCCL/HCCN. It does not
silently fall back to an ambiguous udmac name plus eid_idx.
EOF
}

while (($#)); do
  case "$1" in
    --src-phy) src_phy=$2; shift 2 ;;
    --dst-phy) dst_phy=$2; shift 2 ;;
    --umdk-root) umdk_root=$2; shift 2 ;;
    --urma-include) urma_include=$2; shift 2 ;;
    --urma-lib-dir) urma_lib_dir=$2; shift 2 ;;
    --output-root) output_root=$2; shift 2 ;;
    --route-log) route_log=$2; shift 2 ;;
    --route-discovery-timeout) route_discovery_timeout=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${src_phy}" =~ ^[0-7]$ && "${dst_phy}" =~ ^[0-7]$ && "${src_phy}" != "${dst_phy}" ]] || {
  echo "--src-phy/--dst-phy must be two distinct physical devices in 0..7" >&2
  exit 2
}
[[ "${route_discovery_timeout}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--route-discovery-timeout must be a positive integer" >&2
  exit 2
}

make_args=(UMDK_ROOT="${umdk_root}" URMA_LIB_DIR="${urma_lib_dir}")
[[ -z "${urma_include}" ]] || make_args+=(URMA_INCLUDE="${urma_include}")
make -C "${probe_dir}" "${make_args[@]}"

timestamp=$(date +%Y%m%d_%H%M%S)
run_dir="${output_root}/a5_urma_full_eid_${src_phy}_to_${dst_phy}_${timestamp}"
mkdir -p "${run_dir}"
result_tsv="${run_dir}/full_eid_visibility.tsv"
printf 'route\thop\tside\tphysical_device\teid\tstatus\tresolved_device\tresolved_eid_index\tcontext_verified\n' >"${result_tsv}"

# Ask HCCL for the route inventory of the requested physical pair. The probe
# prints every route before channel acquisition. Select route0 for discovery:
# all candidates are still printed, while the known-good direct channel avoids
# waiting on an unavailable hop-2 channel. No EID bit field is used to infer
# ownership.
discovery_log="${run_dir}/hccl_route_discovery.log"
if [[ -n "${route_log}" ]]; then
  [[ -r "${route_log}" ]] || { echo "cannot read --route-log: ${route_log}" >&2; exit 2; }
  cp "${route_log}" "${discovery_log}"
else
  set +e
  timeout -k 2 "${route_discovery_timeout}" \
    bash "${repo_root}/scripts/run_a5_ccu_urma_route_probe.sh" \
      --devices "${src_phy},${dst_phy}" --route-index 0 --bytes 1024 \
      --warmup 0 --iters 1 --channel-only >"${discovery_log}" 2>&1
  discovery_rc=$?
  set -e
  echo "HCCL route discovery exit status: ${discovery_rc} (timeout/channel failure is acceptable if routes were printed)"
fi

manifest_file="${run_dir}/route_eid_manifest.tsv"
python3 - "${discovery_log}" "${manifest_file}" "${src_phy}" "${dst_phy}" <<'PY'
import re
import sys

log_path, output_path, src_phy, dst_phy = sys.argv[1:]
pattern = re.compile(
    r"\[rank=0 peer=1\]\s+route=(\d+).*?hop=(\d+).*?"
    r"src_phy=(\d+)\s+dst_phy=(\d+).*?"
    r"src_addr=type=\d+,raw=([0-9a-fA-F:]+)\s+"
    r"dst_addr=type=\d+,raw=([0-9a-fA-F:]+)"
)
routes = {}
for line in open(log_path, encoding="utf-8", errors="replace"):
    match = pattern.search(line)
    if not match:
        continue
    route, hop, actual_src, actual_dst, src_eid, dst_eid = match.groups()
    if actual_src == src_phy and actual_dst == dst_phy:
        routes[int(route)] = (int(hop), src_eid.lower(), dst_eid.lower())
if not routes:
    lines = open(log_path, encoding="utf-8", errors="replace").read().splitlines()
    tail = "\n".join(lines[-80:])
    print("===== HCCL route discovery log tail =====", file=sys.stderr)
    print(tail or "<empty log>", file=sys.stderr)
    print("===== end log tail =====", file=sys.stderr)
    raise SystemExit(
        f"no rank0 route inventory for physical {src_phy}->{dst_phy} in {log_path}; "
        "verify the CCU route-probe package and inspect the log"
    )
with open(output_path, "w", encoding="utf-8") as out:
    for route in sorted(routes):
        hop, src_eid, dst_eid = routes[route]
        out.write(f"route{route}\t{hop}\tsrc\t{src_phy}\t{src_eid}\n")
        out.write(f"route{route}\t{hop}\tdst\t{dst_phy}\t{dst_eid}\n")
print(f"Discovered {len(routes)} route(s): {','.join('route'+str(x) for x in sorted(routes))}")
PY
echo "Parsed route manifest:"
cat "${manifest_file}"
manifest=$(cat "${manifest_file}")

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
