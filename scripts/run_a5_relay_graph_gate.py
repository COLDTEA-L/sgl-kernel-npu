#!/usr/bin/env python3
"""Launch two communication ranks; explicitly configured external relays are not ranks."""
import argparse
import csv
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True, type=Path)
    p.add_argument('--devices', required=True, help='ordered physical communication cards, e.g. 2,3')
    p.add_argument('--relay-phys', required=True, help='ordered physical relay cards, e.g. 0,1')
    p.add_argument('--relay-manifest', required=True, type=Path)
    p.add_argument('--notify', choices=['CALLBACK', 'D2H'], required=True)
    p.add_argument('--mode', choices=['GRAPH', 'EAGER'], required=True)
    p.add_argument('--policy', default='cycle', help='cycle, direct, relay0, relay1, mixed')
    p.add_argument('--bytes', type=int, default=4 * 1024 * 1024)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--hold-ms', type=int, default=100)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    try:
        devices = [int(v) for v in a.devices.split(',')]
        relays = [int(v) for v in a.relay_phys.split(',')]
    except ValueError:
        p.error('card IDs must be comma-separated integers')
    if len(devices) != 2 or len(relays) not in (1, 2) or min(devices + relays) < 0 or len(set(devices + relays)) != 2 + len(relays):
        p.error('need two distinct endpoints plus 1/2 distinct external relays')
    if a.bytes < 1024 or a.bytes > 1 << 28 or a.bytes % 1024 or a.iters < 4 or not 0 <= a.hold_ms <= 10000 or a.timeout <= 0:
        p.error('invalid bytes/iters/hold/timeout')
    manifest, binary = a.relay_manifest.resolve(), a.binary.resolve()
    if not binary.is_file() or not manifest.is_file():
        p.error('binary and manifest must exist')
    with manifest.open() as f:
        rows = list(csv.DictReader((line for line in f if not line.startswith('#') and line.strip()), delimiter='\t'))
    try:
        if [int(row['relay_phy']) for row in rows] != relays:
            p.error('manifest relay order differs from --relay-phys')
        for row in rows:
            if int(row['src_phy']) != devices[0] or int(row['dst_phy']) != devices[1]:
                p.error('manifest endpoint order differs from --devices')
    except (ValueError, KeyError):
        p.error('manifest requires valid relay_phy/src_phy/dst_phy')
    choices = {'cycle': None, 'direct': 0, 'mixed': len(relays) + 1}
    choices.update({f'relay{i}': i + 1 for i in range(len(relays))})
    if a.policy not in choices:
        p.error(f'policy must be one of {list(choices)}')
    a.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='a5-relay-graph-gate.', dir=a.output_root))
    env = os.environ.copy()
    env['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(map(str, devices))
    env['HCCL_OP_EXPANSION_MODE'] = 'CCU_SCHED'
    # Avoid process-global routing overrides affecting an explicitly supplied plan.
    for key in list(env):
        if key.startswith(('A5_CCU_SYNTHETIC_', 'A5_CCU_DESC_', 'A5_CCU_PATH_', 'A5_CCU_SOURCE_ROUTE_')) or key in (
                'LD_PRELOAD', 'HCCL_ALGO', 'A5_CCU_CHANNEL_SYNC_MODE', 'A5_GATE_POLICY'):
            env.pop(key)
    if choices[a.policy] is not None:
        env['A5_GATE_POLICY'] = str(choices[a.policy])
    children, logs = [], []
    print(f'RESULT_DIR={out} devices={devices} relay_phys={relays} notify={a.notify} mode={a.mode}', flush=True)
    try:
        for rank in range(2):
            log = (out / f'rank{rank}.log').open('w')
            logs.append(log)
            command = [str(binary), str(rank), str(rank), str(out / 'bootstrap'), str(a.bytes),
                       str(a.iters), str(manifest), a.notify, a.mode, str(a.hold_ms)]
            children.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + a.timeout
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None, 0) for c in children):
                raise RuntimeError(f'worker failed; see {out}')
            if time.monotonic() > deadline:
                raise TimeoutError(f'worker timeout; see {out}')
            time.sleep(0.1)
        if any(c.returncode != 0 for c in children):
            raise RuntimeError(f'worker failure; see {out}')
        for rank in range(2):
            text = (out / f'rank{rank}.log').read_text()
            if f'GRAPH_GATE_RANK_PASS rank={rank} ' not in text or text.count(f'GATE_OP_PASS rank={rank} ') != a.iters * 2:
                raise RuntimeError(f'missing completion accounting rank={rank}; see {out}')
        print('GRAPH_GATE_ALL_RANKS_AND_EXIT_PASS (logical path control; physical relay needs HCCN evidence)', flush=True)
    finally:
        for c in children:
            if c.poll() is None:
                os.killpg(c.pid, signal.SIGTERM)
        for c in children:
            try:
                c.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(c.pid, signal.SIGKILL)
                c.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
