#!/usr/bin/env python3
"""Two-rank launcher and strict max-rank/paired-round summarizer (stdlib only)."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import tempfile
import time


def load_explicit_pair(path, devices):
    data = json.loads(Path(path).read_text())
    if not isinstance(data.get('mapping_note'), str) or not data['mapping_note'].strip():
        raise ValueError('explicit EIDs require mapping_note with physical NPU/die/port mapping evidence')
    eps = data.get('endpoints')
    if not isinstance(eps, list) or len(eps) != 2:
        raise ValueError('exactly two explicit endpoints required, ordered by rank 0,1')
    normalized = []
    for rank, ep in enumerate(eps):
        if (ep.get('rank') != rank or str(ep.get('device')) != devices[rank] or
                type(ep.get('die')) is not int or ep['die'] not in (0, 1)):
            raise ValueError('endpoint rank/device/die does not match visible devices')
        eid = ep.get('eid', '')
        if not isinstance(eid, str) or not re.fullmatch(r'(?:[0-9a-fA-F]{32}|(?:[0-9a-fA-F]{4}:){7}[0-9a-fA-F]{4})', eid):
            raise ValueError('EID must be full 128-bit hex, not a suffix or compressed address')
        eid = eid.replace(':', '').lower()
        if int(eid, 16) == 0:
            raise ValueError('zero EID is not allowed')
        normalized.append(dict(ep, eid=eid))
    if normalized[0]['eid'] == normalized[1]['eid']:
        raise ValueError('local and remote EIDs must differ')
    return dict(data, endpoints=normalized)


def failure_report(root):
    report = {'both_ranks_pass': False, 'ranks': []}
    for rank in range(2):
        p = root / ('rank%d.log' % rank)
        lines = p.read_text(errors='replace').splitlines() if p.exists() else []
        report['ranks'].append({'rank': rank,
            'diagnostics': [x for x in lines if any(tag in x for tag in
                ('L0L1_PHASE', 'L0L1_STAGE', 'L0L1_EXPLICIT_', 'L0L1_PEER_ATTR',
                 'L0L1_RESOLVED', 'BW_COEFF_UNVERIFIABLE', 'L0L1_FAILED'))][-25:]})
    return report


def summarize(directory):
    root = Path(directory)
    options = json.loads((root / 'options.json').read_text())
    if (root / 'exit.status').read_text().strip() != '0':
        raise ValueError('workers did not both exit cleanly')
    for rank in range(2):
        log = (root / ('rank%d.log' % rank)).read_text(errors='replace')
        marker = r'L0L1_RANK_PASS rank=%d suite=%s\b' % (rank, options['suite'])
        if not re.search(marker, log):
            raise ValueError('missing clean completion rank=%d' % rank)
    if options['suite'] not in ('benchmark', 'all'):
        return {'suite': options['suite'], 'both_ranks_pass': True,
                'endpoint_source': 'explicit' if options.get('explicit_eids') else 'graph',
                'note': 'payload checked' if options['suite'] == 'correctness' else 'no payload validation'}
    modes = ['l0', 'l1', 'split'] if options['mode'] == 'all' else [options['mode']]
    expected = set(itertools.product(range(options['rounds']), modes))
    ranks = []
    split0 = options['bytes'] * options['l0_percent'] // 100 // 256 * 256
    for rank in range(2):
        records = {}
        for line in (root / ('rank%d.jsonl' % rank)).read_text().splitlines():
            row = json.loads(line)
            key = (row['round'], row['mode'])
            if key not in expected or key in records or row['rank'] != rank:
                raise ValueError('unexpected/duplicate sample')
            b0 = options['bytes'] if row['mode'] == 'l0' else 0 if row['mode'] == 'l1' else split0
            if (row['bytes_per_direction'] != options['bytes'] or row['l0_bytes'] != b0 or
                    row.get('l1_port_policy', 'any') != options.get('l1_port_policy', 'any') or
                    row.get('endpoint_source', 'graph') != ('explicit' if options.get('explicit_eids') else 'graph') or
                    row['l1_bytes'] != options['bytes'] - b0 or row['timing'] != options['timing'] or
                    not math.isfinite(row['wall_us']) or row['wall_us'] <= 0):
                raise ValueError('sample boundary mismatch')
            records[key] = row
        if set(records) != expected:
            raise ValueError('missing samples')
        ranks.append(records)
    result = {'suite': options['suite'], 'both_ranks_pass': True,
              'l1_port_policy': options.get('l1_port_policy', 'any'),
              'endpoint_source': 'explicit' if options.get('explicit_eids') else 'graph',
              'timing': options['timing'], 'bytes_per_direction': options['bytes'], 'modes': {}}
    for mode in modes:
        samples = [max(ranks[0][r, mode]['wall_us'], ranks[1][r, mode]['wall_us'])
                   for r in range(options['rounds'])]
        median = statistics.median(samples)
        result['modes'][mode] = {'max_rank_us_by_round': samples, 'median_us': median,
                                 'effective_GBps_per_direction': options['bytes'] / median / 1000}
    if len(modes) == 3:
        d = result['modes']
        result['l0_over_split'] = d['l0']['median_us'] / d['split']['median_us']
        result['l1_over_split'] = d['l1']['median_us'] / d['split']['median_us']
        result['note'] = 'Same bytes per direction; ratios >1 favor split. Port mapping still requires hardware counters.'
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summarize', type=Path)
    p.add_argument('--binary', type=Path, default=Path(os.getenv('A5_L0L1_BINARY',
                   '/home/l00934901/profiling/ccu-l0-l1-build/a5_ccu_l0_l1_probe')))
    p.add_argument('--output-root', type=Path, default=Path('/home/l00934901/profiling'))
    p.add_argument('--suite', choices=['endpoint', 'discover', 'create', 'correctness', 'benchmark', 'all'], default='all')
    p.add_argument('--mode', choices=['all', 'l0', 'l1', 'split'], default='all')
    p.add_argument('--timing', choices=['block', 'serial'], default='block')
    p.add_argument('--bytes', type=int, default=8 << 20)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--rounds', type=int, default=6)
    p.add_argument('--l0-percent', type=int, default=50)
    p.add_argument('--l0-index', type=int, default=0)
    p.add_argument('--l1-index', type=int, default=0)
    p.add_argument('--l1-port-policy', choices=['single', 'any'], default='single',
                   help='single requires BW_COEFF=1 at both L1 endpoints; any restores the aggregate experiment')
    p.add_argument('--explicit-eids', type=Path, help='JSON with two verified physical device/EID/die mappings; experimental HcclChannelAcquire override')
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--capture-ports', action='store_true', help='read hccn dev_info/stat before and after workers')
    p.add_argument('--hccn-tool', default='/usr/local/Ascend/driver/tools/hccn_tool')
    return p


def capture_ports(args, root, devices, label):
    if not args.capture_ports:
        return
    with (root / ('ports_%s.log' % label)).open('w') as out:
        for device in devices:
            commands = [[args.hccn_tool, '-g', '-dev_info', '-i', device]]
            # Record all potential ports. An absent die/port is recorded as a
            # command error, never interpreted as link DOWN or zero traffic.
            commands += [[args.hccn_tool, '-g', '-stat', '-i', device, '-u', str(die), '-p', str(port)]
                         for die in range(2) for port in range(9)]
            for cmd in commands:
                out.write('COMMAND ' + json.dumps(cmd) + '\n'); out.flush()
                try:
                    rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, timeout=5, check=False).returncode
                    out.write('EXIT %s\n' % rc)
                except (OSError, subprocess.TimeoutExpired) as e:
                    out.write('CAPTURE_ERROR %s\n' % e)


def run(args):
    devices = os.getenv('ASCEND_RT_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 2 or len(set(devices)) != 2 or not all(x.isdigit() for x in devices):
        raise ValueError('A5_DEVICES must contain two distinct physical device IDs')
    pair = load_explicit_pair(args.explicit_eids, devices) if args.explicit_eids else None
    if pair and (args.mode == 'l0' or args.l1_port_policy != 'any' or args.suite == 'discover'):
        raise ValueError('explicit experiment requires L1, policy any, and endpoint/create/correctness/benchmark/all suite')
    if args.suite == 'endpoint' and (pair is None or args.mode != 'l1'):
        raise ValueError('endpoint suite requires --explicit-eids and --mode l1')
    if not args.binary.is_file():
        raise ValueError('build the probe first: ' + str(args.binary))
    if (args.bytes < 512 or args.bytes > 256 << 20 or args.bytes % 256 or
            not 1 <= args.l0_percent <= 99 or args.bytes * args.l0_percent // 100 // 256 == 0 or
            not 1 <= args.rounds <= 1000 or not 1 <= args.iters <= 1000000 or
            not 0 <= args.warmup <= 1000000 or min(args.l0_index, args.l1_index) < 0 or args.timeout <= 0):
        raise ValueError('invalid size/split/iteration options')
    args.output_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='a5-l0-l1-', dir=str(args.output_root)))
    opts = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if pair:
        (root / 'explicit_eids.json').write_text(json.dumps(pair, indent=2))
    (root / 'options.json').write_text(json.dumps(opts, indent=2))
    env_keys = ['ASCEND_HOME_PATH', 'ASCEND_RT_VISIBLE_DEVICES', 'HCCL_OP_EXPANSION_MODE',
                'HCCL_BUFFSIZE', 'LD_LIBRARY_PATH', 'ASCEND_GLOBAL_LOG_LEVEL', 'ASCEND_SLOG_PRINT_TO_STDOUT']
    (root / 'environment.json').write_text(json.dumps({k: os.getenv(k) for k in env_keys}, indent=2))
    for source in ['/etc/hccl_rootinfo.json', '/usr/local/Ascend/driver/topo/950/atlas_950_1.json']:
        if Path(source).is_file():
            shutil.copyfile(source, root / Path(source).name)
    repo = Path(__file__).resolve().parents[1]
    with (root / 'git.txt').open('w') as f:
        subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(repo), stdout=f, stderr=f, check=False)
        subprocess.run(['git', 'status', '--short'], cwd=str(repo), stdout=f, stderr=f, check=False)
    print('OUTPUT_DIR:', root, flush=True)
    capture_ports(args, root, devices, 'before')
    workers, files = [], []
    codes = []
    try:
        for rank in range(2):
            cmd = [str(args.binary.resolve()), '--rank', str(rank), '--device', str(rank), '--dir', str(root)]
            for key in ['suite', 'mode', 'timing', 'bytes', 'warmup', 'iters', 'rounds', 'l0_percent', 'l0_index', 'l1_index', 'l1_port_policy']:
                cmd += ['--' + key.replace('_', '-'), str(getattr(args, key))]
            if pair:
                for ep in pair['endpoints']:
                    cmd += ['--eid%d' % ep['rank'], ep['eid'], '--eid-die%d' % ep['rank'], str(ep['die'])]
            (root / ('command%d.json' % rank)).write_text(json.dumps(cmd))
            f = (root / ('rank%d.log' % rank)).open('w'); files.append(f)
            workers.append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + args.timeout
        while True:
            codes = [p.poll() for p in workers]
            if any(c not in (None, 0) for c in codes) or all(c is not None for c in codes):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('worker deadline exceeded')
            time.sleep(0.2)
    finally:
        # Only these worker process groups; never pkill other users' workloads.
        for p in workers:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in workers:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
        for f in files:
            f.close()
        (root / 'worker_exit.json').write_text(json.dumps([p.returncode for p in workers]))
        # A timeout/exception may terminate children with 0: only observed clean exits count.
        (root / 'exit.status').write_text('0\n' if codes == [0, 0] else '1\n')
        if codes != [0, 0]:
            (root / 'failure_summary.json').write_text(json.dumps(failure_report(root), indent=2))
    capture_ports(args, root, devices, 'after')
    if codes != [0, 0]:
        report = failure_report(root)
        (root / 'failure_summary.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError('L0L1_WORKERS_FAILED: ' + str(root))
    summary = summarize(root)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('L0L1_ALL_RANKS_AND_EXIT_PASS', root)


if __name__ == '__main__':
    args = parser().parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.summarize), indent=2))
    else:
        run(args)
