#!/usr/bin/env python3
"""Summarize paired host-boundary samples from successful EID combination audits."""
import argparse
import json
import math
from pathlib import Path
import statistics
from summarize_a5_ccu_eid import load


def summarize_records(records):
    groups = {}
    for rank, row in records:
        samples = row['samples']
        n = row['iterations']
        if n <= 0 or len(samples) != n:
            raise ValueError('missing measured iteration samples')
        fields = ('counts_host_us', 'action_host_us', 'fence_host_us', 'call_host_us')
        vals = [row['elapsed_host_us'], row['final_drain_host_us']] + [s[k] for s in samples for k in fields]
        if any(not math.isfinite(v) or v < 0 for v in vals):
            raise ValueError('invalid timing sample')
        means = {k: statistics.mean(s[k] for s in samples) for k in fields}
        result = dict(rank=rank, elapsed_us=row['elapsed_host_us'], counts_us=means['counts_host_us'],
                      action_us=means['action_host_us'], fence_us=means['fence_host_us'],
                      drain_per_call_us=row['final_drain_host_us']/n)
        result['loop_other_us'] = result['elapsed_us'] - means['call_host_us'] - result['drain_per_call_us']
        key = row['metric'], row['mode'], row['round']
        if key not in groups or result['elapsed_us'] > groups[key]['elapsed_us']:
            groups[key] = result  # same rank for all components, not separate maxima
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    args = p.parse_args()
    first, _ = load(args.run)  # requires all 8 rank completion markers and exit.status=0
    if not first['options'].get('combination_audit', False):
        raise ValueError('run was not measured with --combination-audit')
    results = [json.loads((args.run/'results'/f'rank{r}.json').read_text()) for r in range(8)]
    print('HOST_BOUNDARIES_ONLY: counts/action may include waiting; not device kernel durations')
    print('Each round selects one slowest rank; displayed columns are medians across rounds.')
    for i, report in enumerate(first['reports']):
        records = [(r, row) for r in range(8) for row in results[r]['reports'][i]['rank_local_audit']]
        grouped = summarize_records(records)
        expected = {(metric, mode, turn) for metric, modes in report['samples_us'].items()
                    if metric.startswith('audit_') for mode in modes for turn in range(first['options']['rounds'])}
        if set(grouped) != expected or len(records) != 8*len(expected):
            raise ValueError('missing audit rank/round data')
        for key in expected:
            ranks = [r for r,row in records if (row['metric'],row['mode'],row['round']) == key]
            if sorted(ranks) != list(range(8)): raise ValueError('duplicate or missing rank')
        print(f"case={report['case']} dtype={first['options']['dtype']} counts_path={first['options'].get('counts_path','legacy')}")
        print('metric mode elapsed_us counts_us action_us fence_us drain_per_call_us loop_other_us')
        for metric, mode in sorted({k[:2] for k in grouped}):
            rows = [v for k,v in grouped.items() if k[:2] == (metric,mode)]
            fields = ('elapsed_us','counts_us','action_us','fence_us','drain_per_call_us','loop_other_us')
            print(metric, mode, *[f'{statistics.median(r[f] for r in rows):.2f}' for f in fields])
    print('EP8_COMBINATION_AUDIT_SUMMARY_PASS')


if __name__ == '__main__': main()
