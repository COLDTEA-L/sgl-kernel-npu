#!/usr/bin/env python3
"""Require eight completed workers and compare direct/multipath with identical input."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(root):
    root = Path(root)
    if (root / 'exit.status').read_text().strip() != '0':
        raise ValueError('worker launch did not exit successfully')
    ranks = [json.loads((root / 'results' / f'rank{r}.json').read_text()) for r in range(8)]
    first = ranks[0]
    for r, result in enumerate(ranks):
        if result['rank'] != r or result['ranks'] != 8 or result['correctness'] != 'PASS':
            raise ValueError('missing rank or correctness failure')
        for key in ('mode', 'dtype', 'width', 'backend', 'timing_scope', 'dynamic_scope', 'benchmark'):
            if result[key] != first[key]: raise ValueError(f'inconsistent {key}')
        if [c['case'] for c in result['cases']] != [c['case'] for c in first['cases']]:
            raise ValueError('different case order')
    output = {k: first[k] for k in ('dtype', 'width', 'mode', 'backend', 'timing_scope', 'dynamic_scope', 'benchmark')}
    output['correctness'] = 'PASS'; output['cases'] = []
    for index, case in enumerate(first['cases']):
        all_cases = [r['cases'][index] for r in ranks]
        if any(c['correctness'] != 'PASS' or c['fingerprint'] != case['fingerprint'] for c in all_cases):
            raise ValueError('plan/correctness differs across ranks')
        modes = set(case['prepared_round_us'])
        if any(set(c['prepared_round_us']) != modes or set(c['dynamic_mean_us']) != modes for c in all_cases):
            raise ValueError('inconsistent measured modes')
        summary = dict(case=case['case'], plan=case['plan'], schedule=case['schedule'],
                       planner_max_rank_us=max(c['planner_us'] for c in all_cases), prepared={}, dynamic={})
        for mode in sorted(modes):
            rounds = len(case['prepared_round_us'][mode])
            if not rounds or any(len(c['prepared_round_us'][mode]) != rounds for c in all_cases):
                raise ValueError('missing timing rounds')
            values = [max(c['prepared_round_us'][mode][i] for c in all_cases) for i in range(rounds)]
            dynamic = max(c['dynamic_mean_us'][mode] for c in all_cases)
            if any(not math.isfinite(v) or v <= 0 for v in values + [dynamic]):
                raise ValueError('invalid timing value')
            summary['prepared'][mode] = dict(slowest_rank_round_us=values, mean_us=statistics.mean(values))
            summary['dynamic'][mode] = dict(mean_us=dynamic)
        if modes == {'direct', 'multipath'}:
            for scope in ('prepared', 'dynamic'):
                summary[scope]['direct_over_multipath'] = (summary[scope]['direct']['mean_us'] /
                                                          summary[scope]['multipath']['mean_us'])
            summary['analysis'] = ('dynamic multipath faster on this workload' if
                                   summary['dynamic']['direct_over_multipath'] > 1 else
                                   'no dynamic speedup; retain direct for this workload')
        output['cases'].append(summary)
    return output


def compare_runs(direct_root, multipath_root):
    direct, multi = summarize(direct_root), summarize(multipath_root)
    for key in ('dtype', 'width', 'backend', 'timing_scope', 'dynamic_scope', 'benchmark'):
        if direct[key] != multi[key]: raise ValueError(f'comparison differs in {key}')
    if [c['case'] for c in direct['cases']] != [c['case'] for c in multi['cases']]:
        raise ValueError('comparison has different cases')
    result = []
    for d, m in zip(direct['cases'], multi['cases']):
        if d['plan']['fingerprint'] != m['plan']['fingerprint']:
            raise ValueError('comparison counts/row bytes/relay plan differ')
        row = dict(case=d['case'])
        for scope in ('prepared', 'dynamic'):
            dt, mt = d[scope]['direct']['mean_us'], m[scope]['multipath']['mean_us']
            row[scope] = dict(direct_us=dt, multipath_us=mt, direct_over_multipath=dt / mt)
        result.append(row)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path, nargs='?')
    p.add_argument('--compare', type=Path, nargs=2, metavar=('DIRECT_RUN', 'MULTIPATH_RUN'))
    args = p.parse_args()
    if args.compare:
        comparison = compare_runs(*args.compare)
        (args.compare[1] / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        print(json.dumps(comparison, indent=2))
        return
    if args.run_dir is None: p.error('run_dir or --compare is required')
    summary = summarize(args.run_dir)
    (args.run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('EP8_HEURISTIC_ALL_RANKS_AND_EXIT_PASS')
    for c in summary['cases']:
        print(json.dumps({k: c[k] for k in ('case', 'planner_max_rank_us', 'prepared', 'dynamic')}))
        if 'analysis' in c: print(c['analysis'])
    print('summary: ' + str(args.run_dir / 'summary.json'))


if __name__ == '__main__':
    main()
