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


def summarize_stage_records(records, grouped):
    """Choose the same slowest block rank as outer accounting; retain tails/skew."""
    buckets = {}
    for rank, row in records:
        key = row['metric'], row['mode'], row['round']
        samples = row['samples']
        if any(not s.get('stages') or not math.isfinite(s.get('start_host_s', float('nan')))
               for s in samples):
            raise ValueError('missing stage samples or host timestamps')
        fields = set(samples[0]['stages'])
        for sample in samples:
            if set(sample['stages']) != fields:
                raise ValueError('inconsistent stage fields inside block')
            if any(not math.isfinite(v) or v < 0 for v in sample['stages'].values()):
                raise ValueError('invalid stage timing')
        buckets.setdefault(key, {})[rank] = row
    output = []
    for key in sorted(buckets):
        by_rank = buckets[key]
        if set(by_rank) != set(range(8)):
            raise ValueError('stage summary requires all eight ranks')
        chosen = grouped[key]
        samples = by_rank[chosen['rank']]['samples']
        if any(len(row['samples']) != len(samples) for row in by_rank.values()):
            raise ValueError('different stage iteration counts')
        avg = {field: statistics.mean(s['stages'][field] for s in samples)
               for field in samples[0]['stages']}
        spread = [(max(row['samples'][i]['start_host_s'] for row in by_rank.values()) -
                   min(row['samples'][i]['start_host_s'] for row in by_rank.values())) * 1e6
                  for i in range(len(samples))]
        calls = sorted(s['call_host_us'] for s in samples)
        tail = dict(call_p50_us=statistics.median(calls),
                    call_p95_us=calls[max(0, math.ceil(len(calls)*.95)-1)],
                    call_max_us=max(calls),
                    start_spread_mean_us=statistics.mean(spread), start_spread_max_us=max(spread))
        output.append(dict(metric=key[0], mode=key[1], round=key[2], **chosen,
                           stages_mean_us=avg, **tail))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output-json', type=Path, help='paired per-round stage details (optional)')
    args = p.parse_args()
    first, _ = load(args.run)  # requires all 8 rank completion markers and exit.status=0
    if not first['options'].get('combination_audit', False):
        raise ValueError('run was not measured with --combination-audit')
    results = [json.loads((args.run/'results'/f'rank{r}.json').read_text()) for r in range(8)]
    print('HOST_BOUNDARIES_ONLY: counts/action may include waiting; not device kernel durations')
    print('Each round selects one slowest rank; displayed columns are medians across rounds.')
    detailed_reports = []
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
        if first['options'].get('dynamic_stage_audit', False):
            details = summarize_stage_records(records, grouped)
            detailed_reports.append(dict(case=report['case'], rounds=details))
            print('STAGE_ROUND: paired slow-rank components; cpu_us = calling thread only; start spread = single-host clocks')
            for row in details:
                print('STAGE_ROUND ' + json.dumps(row, sort_keys=True))
    if args.output_json:
        if not detailed_reports:
            raise ValueError('--output-json requires --dynamic-stage-audit data')
        args.output_json.write_text(json.dumps(dict(run=str(args.run), reports=detailed_reports), indent=2) + '\n')
    print('EP8_COMBINATION_AUDIT_SUMMARY_PASS')


if __name__ == '__main__': main()
