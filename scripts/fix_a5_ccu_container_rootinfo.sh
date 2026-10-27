#!/usr/bin/env bash
# Run on the experiment HOST. Check current-machine addresses before installing.
set -Ee -o pipefail
container=sglang_yuanwen_old
repo=/home/l00934901/sgl-kernel-npu
shared_root=/home/l00934901
root_info=
src=2
dst=3
relay=4
relay_die=1
apply=0
machine_args=(--query-timeout 15)
usage() {
    cat <<'HELP'
Run on HOST: bash scripts/fix_a5_ccu_container_rootinfo.sh [--apply]
  --container NAME       default sglang_yuanwen_old
  --repo PATH            path shared with container, default /home/l00934901/sgl-kernel-npu
  --shared-root PATH     staging directory parent, default /home/l00934901
  --root-info PATH       host file; auto-detect /etc/hccl_root[iI]nfo.json
  --src-phy ID --dst-phy ID --relay-phy ID --relay-die 0|1|all
                         defaults 2,3,4 and relay die 1
  --topology-json PATH --hccn-tool PATH --query-timeout SECONDS
  --net-layer N --net-instance-id LABEL
                         paths/options interpreted INSIDE container
Default: stage + read-only address/port checks; container /etc stays unchanged.
--apply: after successful checks, back up both container rootinfo names and
         replace them with the verified host snapshot. Does not add bind mounts.
HELP
}
while (($#)); do
    case "$1" in
        --apply) apply=1; shift;;
        --container) container=$2; shift 2;;
        --repo) repo=$2; shift 2;;
        --shared-root) shared_root=$2; shift 2;;
        --root-info) root_info=$2; shift 2;;
        --src-phy) src=$2; shift 2;;
        --dst-phy) dst=$2; shift 2;;
        --relay-phy) relay=$2; shift 2;;
        --relay-die) relay_die=$2; shift 2;;
        --topology-json|--hccn-tool|--query-timeout|--net-layer|--net-instance-id)
            machine_args+=("$1" "$2"); shift 2;;
        --help|-h) usage; exit 0;;
        *) echo "Unsupported argument: $1" >&2; exit 2;;
    esac
done
for id in "$src" "$dst" "$relay"; do
    [[ "$id" =~ ^[0-9]+$ ]] || { echo 'Physical IDs must be nonnegative integers' >&2; exit 2; }
done
[[ "$src" != "$dst" && "$src" != "$relay" && "$dst" != "$relay" ]] || {
    echo 'Source, destination and relay must be distinct' >&2; exit 2;
}
case "$relay_die" in 0|1|all) ;; *) echo 'Invalid --relay-die' >&2; exit 2;; esac
if [[ -z "$root_info" ]]; then
    candidates=()
    for path in /etc/hccl_rootinfo.json /etc/hccl_rootInfo.json; do
        [[ ! -f "$path" ]] || candidates+=("$path")
    done
    ((${#candidates[@]} > 0)) || { echo 'No host rootinfo; specify --root-info PATH' >&2; exit 2; }
    root_info=${candidates[0]}
    for path in "${candidates[@]}"; do
        cmp -s "$root_info" "$path" || {
            echo 'Host rootinfo names differ; explicitly select --root-info PATH' >&2; exit 2;
        }
    done
fi
[[ -f "$root_info" && -r "$root_info" ]] || { echo "Unreadable host file: $root_info" >&2; exit 2; }
[[ "$shared_root" == /* && "$repo" == /* && -d "$shared_root" ]] || {
    echo 'Absolute, existing shared directory/repo paths are required' >&2; exit 2;
}
[[ "$(docker inspect --format '{{.State.Running}}' "$container")" == true ]] || {
    echo "Container is not running: $container" >&2; exit 2;
}
fix_dir=$(mktemp -d "${shared_root%/}/a5-rootinfo-fix.XXXXXX")
trap 'echo "ROOTINFO_FIX_FAILED line=${LINENO}; evidence/backup directory: ${fix_dir}" >&2' ERR
cp -- "$root_info" "$fix_dir/host-rootinfo.json"
chmod 444 "$fix_dir/host-rootinfo.json"
echo "Evidence/backup directory: $fix_dir"
sha256sum "$root_info" "$fix_dir/host-rootinfo.json"
docker exec "$container" test -r "$fix_dir/host-rootinfo.json"
# All queries run inside the container with its actual driver mapping. No HCCL
# communicator, NPU tensor allocation or CCU kernel is created by this check.
docker exec -i -w "$repo" "$container" python3 - \
    "$fix_dir" "$src" "$dst" "$relay" "$relay_die" "${machine_args[@]}" <<'PY'
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from a5_ccu_machine_topology import add_machine_arguments, load_machine, directed_links
folder = Path(sys.argv[1])
src, dst, relay = map(int, sys.argv[2:5])
if len({src, dst, relay}) != 3:
    raise SystemExit('Source, destination and relay physical IDs must be distinct')
die = sys.argv[5]
p = argparse.ArgumentParser()
add_machine_arguments(p)
a = p.parse_args(['--root-info', str(folder / 'host-rootinfo.json'),
                  '--diagnostics-json', str(folder / 'host-rootinfo-check.json'), *sys.argv[6:]])
entries, edges, _ = load_machine(a, {src, dst, relay})
valid = []
for left in directed_links(edges, src, relay, a.net_layer):
    for right in directed_links(edges, dst, relay, a.net_layer):
        rd, _ = left['relay_port']
        if rd != right['relay_port'][0] or (die != 'all' and rd != int(die)):
            continue
        endpoints = [(src, left['endpoint_port']), (relay, left['relay_port']),
                     (relay, right['relay_port']), (dst, right['endpoint_port'])]
        matches = []
        for device, (ed, port) in endpoints:
            bindings = [b for b in entries if (b['device'], b['die'], b['port']) == (device, ed, port)]
            print('HOST_BINDING', device, f'{ed}/{port}', json.dumps([
                {key: b[key] for key in ('eid', 'plane_id', 'live_eid_present', 'port_state', 'verified')}
                for b in bindings]), flush=True)
            matches.append(len(bindings) == 1 and bindings[0]['verified'])
        if all(matches):
            valid.append(endpoints)
if not valid:
    raise SystemExit('HOST_ROOTINFO_CHECK_FAILED: host addresses/ports do not match the live topology; '
                     'container /etc was not modified')
print('HOST_ROOTINFO_CHECK_PASS: physical endpoint addresses verified; not a ChannelAcquire/NPU test', flush=True)
PY
# Refuse to install a snapshot if the host file changed while it was checked.
cmp -s "$root_info" "$fix_dir/host-rootinfo.json" || {
    echo 'Host rootinfo changed during validation; retry from a new snapshot' >&2; exit 1;
}
if ((apply == 0)); then
    echo 'Check complete. Container /etc unchanged. Re-run with --apply to install a newly verified snapshot.'
    exit 0
fi
# Never rename over a runtime/bind-mounted file or write through a mounted /etc.
docker exec -i "$container" python3 - <<'PY'
from pathlib import Path
import os
if os.geteuid() != 0:
    raise SystemExit('--apply requires container root')
protected = {'/etc', '/etc/hccl_rootinfo.json', '/etc/hccl_rootInfo.json'}
for line in Path('/proc/self/mountinfo').read_text().splitlines():
    if line.split()[4] in protected:
        raise SystemExit('rootinfo or /etc is already mounted; correct its mount source when recreating '
                         'the container, rather than overwriting the mounted file')
PY
for name in hccl_rootinfo.json hccl_rootInfo.json; do
    if docker exec "$container" test -e "/etc/$name"; then
        docker exec "$container" test -f "/etc/$name"
        docker cp -L "$container:/etc/$name" "$fix_dir/container-$name.before"
    else
        touch "$fix_dir/container-$name.absent"
    fi
done
# Both spellings receive identical data, so consumers using either name and
# auto-discovery cannot accidentally select an old image copy.
docker exec -i "$container" bash -s -- "$fix_dir/host-rootinfo.json" <<'BASH'
set -Ee -o pipefail
source_file=$1
temporary=
trap '[[ -z "$temporary" ]] || rm -f -- "$temporary"' EXIT
for name in hccl_rootinfo.json hccl_rootInfo.json; do
    temporary=$(mktemp /etc/.a5-rootinfo.XXXXXX)
    cp -- "$source_file" "$temporary"
    chmod 644 "$temporary"
    mv -f -- "$temporary" "/etc/$name"
    temporary=
done
BASH
expected=$(sha256sum "$fix_dir/host-rootinfo.json")
expected=${expected%% *}
for name in hccl_rootinfo.json hccl_rootInfo.json; do
    actual=$(docker exec "$container" sha256sum "/etc/$name")
    echo "$actual"
    [[ "${actual%% *}" == "$expected" ]] || { echo "Installed hash mismatch: $name" >&2; exit 1; }
done
echo "ROOTINFO_APPLY_PASS: backups in $fix_dir; host file unchanged."
echo 'This synchronized container files; persistent read-only mounts are needed when creating its replacement.'
echo 'Re-run routing checks and communication initialization in new test processes; NPU correctness is not yet verified.'
