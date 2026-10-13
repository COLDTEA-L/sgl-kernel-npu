#!/usr/bin/env python3
"""CCU task time != host batch latency. Missing/extra profiling never becomes PASS."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics


def positive(x):
    return type(x) in (int, float) and math.isfinite(x) and x > 0


def norm(s):
    return re.sub('[^a-z0-9]', '', str(s).lower())


def parse_device(root, pid):
    """Prefer explicit-us hardware task CSVs, else complete Chrome Ccu events.

    Never count API call names such as HcclCcuKernelLaunch/rtCcuLaunch. Never
    interpret size(Byte) or host API duration as the local-copy byte count/time.
    """
    records, copies, sources = {}, {}, []
    for file in sorted(root.rglob('task_time*.csv')):
        with file.open(encoding='utf-8-sig', newline='') as f:
            for index, raw in enumerate(csv.DictReader(f)):
                row = {norm(k): v for k, v in raw.items() if k}
                kind = norm(row.get('tasktype', row.get('type', '')))
                if kind != 'ccu' and 'memcpy' not in kind:
                    continue
                owner = row.get('pid', row.get('processid', ''))
                if owner and str(owner) != str(pid):
                    continue
                duration = row.get('taskdurationus', row.get('durationus'))
                try:
                    duration = float(duration)
                except (TypeError, ValueError):
                    continue  # unknown unit/schema: fail closed rather than guessing
                if not positive(duration):
                    continue
                start = row.get('starttimeus', row.get('taskstarttimeus', row.get('starttime', '')))
                key = (row.get('deviceid', ''), owner, row.get('streamid', ''),
                       row.get('taskid', ''), start, duration)
                if not start:
                    key += (str(file), index)  # no reliable cross-file dedup key
                target = records if kind == 'ccu' else copies
                target[key] = dict(duration_us=duration, source=str(file), format='task_time_csv')
                sources.append(str(file))
    if records:
        return list(records.values()), len(copies), sorted(set(sources))
    for file in sorted(root.rglob('msprof*.json')):
        try:
            data = json.loads(file.read_text())
            events = data.get('traceEvents', []) if isinstance(data, dict) else data
            if not isinstance(events, list):
                continue
            for index, event in enumerate(events):
                if not isinstance(event, dict) or event.get('ph') != 'X':
                    continue
                name = norm(event.get('name', ''))
                if name != 'ccu' and name not in ('memcpy', 'memcpyasync', 'memcpyasyn'):
                    continue
                # API/host layers must not masquerade as hardware tasks.
                category = norm(event.get('cat', ''))
                if any(x in category for x in ('host', 'api', 'cann')):
                    continue
                duration = event.get('dur')  # Chrome complete-event duration is microseconds
                if not positive(duration):
                    continue
                args = event.get('args') or {}
                if not isinstance(args, dict):
                    continue
                owner = args.get('pid', args.get('processId'))
                if owner is not None and str(owner) != str(pid):
                    continue
                key = (event.get('pid'), event.get('tid'), event.get('ts'), duration,
                       args.get('taskId', args.get('task_id')))
                if event.get('ts') is None:
                    key += (str(file), index)
                target = records if name == 'ccu' else copies
                target[key] = dict(duration_us=duration, source=str(file), format='chrome_us')
                sources.append(str(file))
        except (OSError, ValueError, TypeError):
            continue
    return list(records.values()), len(copies), sorted(set(sources))


def analyze(root):
    settings = json.loads((root / 'run_settings.json').read_text())
    proof = json.loads((root / 'local_copy_summary.json').read_text())
    functional = proof.get('complete') is True and settings.get('benchmark') is True and not settings.get('builder_only')
    rows, samples, host_ok, device_ok = [], [], True, True
    for case in settings['cases']:
        directory = root / 'cases' / case['name']
        for rank, physical in enumerate(settings['devices']):
            row = dict(case=case['name'], rank=rank, physical_device=physical,
                copy_mode=case['copy_mode'], bytes=case['bytes'], host_batch_avg_us=None,
                group_slice_bytes=case.get('group_slice_bytes',0),
                group_lanes=case.get('group_lanes',0),
                group_ms_budget=case.get('group_ms_budget',0),
                ccu_samples=0, expected_samples=settings['iterations'], ccu_min_us=None,
                ccu_median_us=None, ccu_mean_us=None, ccu_max_us=None, payload_GB_s=None,
                nominal_read_write_GB_s=None, formal_memcpy_tasks=0, result='INCOMPLETE', reason='')
            try:
                log = (directory / f'rank{rank}.log').read_text()
                events = [json.loads(line.split(' ', 1)[1]) for line in log.splitlines()
                          if line.startswith('LOCAL_COPY_PERF_JSON ')]
                if len(events) != 1 or not isinstance(events[0], dict):
                    raise ValueError('missing/duplicate performance result')
                event = events[0]
                if 'group_slice_bytes' in case:
                    geometry = dict(group_slice_bytes=case['group_slice_bytes'],group_lanes=case.get('group_lanes',8))
                    if 'group_ms_budget' in case: geometry['group_ms_budget']=case['group_ms_budget']
                    for key,expected in geometry.items():
                        if type(event.get(key)) is not int or event[key] != expected:
                            raise ValueError(f'wrong/missing {key}; do not compare mislabeled slice timings')
                for key, expected in (('rank', rank), ('bytes', case['bytes']),
                    ('warmup', settings['warmup']), ('iterations', settings['iterations'])):
                    if type(event.get(key)) is not int or event[key] != expected:
                        raise ValueError(f'bad {key}')
                if (event.get('copy_mode') != case['copy_mode'] or event.get('engine') != 'CCU' or
                    event.get('post_check') != 'PASS' or event.get('formal_h2d_d2h') is not False or
                    event.get('remote_write') is not False or
                    event.get('profile_requested') is not settings['profile'] or
                    type(event.get('pid')) is not int or event['pid'] < 1 or
                    not positive(event.get('host_batch_total_us')) or not positive(event.get('host_batch_avg_us')) or
                    not math.isclose(event['host_batch_avg_us'], event['host_batch_total_us']/settings['iterations'], rel_tol=1e-6)):
                    raise ValueError('invalid timing boundary/result')
                row['host_batch_avg_us'] = event['host_batch_avg_us']
                if not settings['profile']:
                    row['result'] = 'HOST_ONLY'; device_ok = False
                else:
                    records, memcpy, sources = parse_device(directory / 'profiling' / f'rank{rank}', event['pid'])
                    row['ccu_samples'], row['formal_memcpy_tasks'] = len(records), memcpy
                    if len(records) != settings['iterations'] or memcpy:
                        device_ok = False
                        row['reason'] = f'expected {settings["iterations"]} CCU and 0 memcpy; got {len(records)} and {memcpy}; inspect export.log/raw PROF'
                    else:
                        values = [r['duration_us'] for r in records]
                        row.update(ccu_min_us=min(values), ccu_median_us=statistics.median(values),
                            ccu_mean_us=statistics.mean(values), ccu_max_us=max(values),
                            payload_GB_s=case['bytes']/(statistics.mean(values)*1000),
                            nominal_read_write_GB_s=2*case['bytes']/(statistics.mean(values)*1000), result='DEVICE_TIMING_CAPTURED')
                        for index, record in enumerate(records):
                            samples.append(dict(case=case['name'], rank=rank, sample=index, **record))
            except (OSError, ValueError, TypeError, KeyError) as error:
                host_ok = False; device_ok = False; row['reason'] = str(error)
            rows.append(row)
    complete = functional and host_ok and (device_ok if settings['profile'] else True)
    conclusion = ('DEVICE_TIMING_CAPTURED' if complete and settings['profile'] else
                  'FUNCTIONAL_HOST_ONLY' if complete else 'INCOMPLETE_OR_FAIL')
    summary = dict(complete=complete, conclusion=conclusion, functional_pass=functional,
        devices=settings['devices'], profile_requested=settings['profile'], cases=rows)
    with (root / 'local_copy_perf_summary.tsv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['case'], delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    with (root / 'local_copy_device_samples.tsv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case','rank','sample','duration_us','source','format'], delimiter='\t')
        writer.writeheader(); writer.writerows(samples)
    (root / 'local_copy_perf_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    report = ['# Isolated CCU local-copy performance', '', f'- conclusion: **{conclusion}**',
        f'- functional correctness + queued snapshots + guards + cleanup: `{functional}`',
        f'- warmup: `{settings["warmup"]}`; formal calls: `{settings["iterations"]}`; CPU sync outside both loops.',
        '- Registration/allocation/input preparation/readback are excluded; formal body is local CCU launches only.', '',
        '| case | rank/physical | host batch avg us | CCU count | CCU min/median/mean/max us | payload GB/s | status |',
        '|---|---|---:|---:|---|---:|---|']
    for r in rows:
        durations='/'.join('NA' if r[k] is None else f'{r[k]:.3f}' for k in ('ccu_min_us','ccu_median_us','ccu_mean_us','ccu_max_us'))
        report.append(f'| {r["case"]} | {r["rank"]}/{r["physical_device"]} | {r["host_batch_avg_us"]} | {r["ccu_samples"]} | {durations} | {r["payload_GB_s"]} | {r["result"]} |')
    report += ['', 'Host batch time includes submission/queue bubbles/final synchronization: NOT pure device copy latency.',
        'CCU task duration includes this micro-kernel control/WaitEvent overhead; no remote Channel/peer waits.',
        'Payload GB/s = actual configured bytes / mean CCU us; nominal read+write GB/s is twice this, NOT measured HBM counters.',
        'Ignore profiling size(Byte) sentinels such as 4294967295. Do not substitute host latency if device samples are missing.',
        'LocalCopyNb throughput is not automatically the device aggregate HBM bandwidth.',
        'chunked is serial 4KiB + WaitEvent, NOT native GroupCopy or a parallel pipeline.', '', '## Diagnostics']
    report += [f'- {r["case"]} rank{r["rank"]}: {r["reason"]}' for r in rows if r['reason']]
    (root / 'local_copy_perf_report.md').write_text('\n'.join(report)+'\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir', type=Path, required=True)
    root = p.parse_args().run_dir
    summary = analyze(root)
    print(root / 'local_copy_perf_report.md')
    print((root / 'local_copy_perf_report.md').read_text())
    raise SystemExit(0 if summary['complete'] else 1)
