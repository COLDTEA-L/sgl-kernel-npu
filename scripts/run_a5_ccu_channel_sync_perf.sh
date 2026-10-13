#!/usr/bin/env bash
# Sequential, process-isolated A/B/C. No global routes, drivers or firmware writes.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
devices=6,7
relays=0,1,2,3,4,5
bytes=4194304
warmup=100
iters=20
repeats=1
profile=1
output_root=/home/l00934901/profiling
extra=()
while (($#)); do
    case "$1" in
        --devices) devices=$2; shift 2;;
        --relay-phys) relays=$2; shift 2;;
        --bytes) bytes=$2; shift 2;;
        --warmup) warmup=$2; shift 2;;
        --iters) iters=$2; shift 2;;
        --repeats) repeats=$2; shift 2;;
        --output-root) output_root=$2; shift 2;;
        --no-profile) profile=0; shift;;
        --profile) profile=1; shift;;
        --dtype|--cann-root|--timeout-seconds|--topology|--topology-json|--direct-route|--available-phys)
            extra+=("$1" "$2"); shift 2;;
        --help)
            echo 'A/B/C: baseline, packed, packed-ready; default compute 6,7 + explicit relays 0..5.'
            echo 'Uses self-copy ccu, 32KiB slices, batch sync, no graph capture; warmup100 + formal20, one repeat, profile enabled.'
            echo 'Override --devices/--relay-phys, --bytes/--dtype, --warmup/--iters/--repeats, --no-profile, --output-root.'
            exit 0;;
        *) echo "unsupported argument: $1" >&2; exit 2;;
    esac
done
for value in "$bytes" "$iters" "$repeats"; do
    [[ "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]] || exit 2
done
[[ "$warmup" =~ ^[0-9]+$ ]] || exit 2
mkdir -p "$output_root"
root=$(mktemp -d "${output_root}/a5_ccu_channel_sync_${devices//,/_}_$(date +%Y%m%d_%H%M%S).XXXXXX")
printf 'mode\tstatus\trun_dir\n' > "$root/runs.tsv"
echo "A/B/C result directory: $root"
failed=0
profile_args=(); ((profile == 0)) || profile_args=(--profile)
for mode in baseline packed packed-ready; do
    mkdir -p "$root/$mode"
    set +e
    bash "$repo/scripts/run_a5_ccu_peer_plan_alltoall.sh" \
        --devices "$devices" --relay-phys "$relays" --bytes "$bytes" \
        --self-copy-mode ccu --channel-sync-mode "$mode" \
        --sync-mode batch --graph-backend none --warmup "$warmup" --iters "$iters" \
        --repeats "$repeats" --output-root "$root/$mode" \
        "${extra[@]}" "${profile_args[@]}" > "$root/$mode.log" 2>&1
    rc=$?
    set -e
    # This is an owned fresh output directory, not a search across other runs.
    runs=("$root/$mode"/a5_ccu_peer_plan_*)
    result_dir=
    if ((${#runs[@]} == 1)) && [[ -d "${runs[0]}" ]]; then result_dir=${runs[0]}; fi
    printf '%s\t%s\t%s\n' "$mode" "$rc" "$result_dir" >> "$root/runs.tsv"
    echo "$mode: status=$rc ($root/$mode.log)"
    if ((rc != 0)); then
        tail -n 60 "$root/$mode.log"
        failed=1
        echo 'Stopped after failure; do not benchmark a partly validated protocol.'
        break
    fi
done
python3 "$repo/scripts/analyze_a5_ccu_channel_sync.py" --run-dir "$root"
sed -n '1,220p' "$root/channel_sync_report.md"
echo "A/B/C result directory: $root"
exit "$failed"
