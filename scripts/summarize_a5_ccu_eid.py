#!/usr/bin/env python3
"""Summarize completed EID runs; optionally compare separate native/multipath runs."""
import argparse
import json
from pathlib import Path
import re


def load(root):
    if (root / 'exit.status').read_text().strip() != '0': raise ValueError('run did not exit successfully')
    results = [json.loads((root/'results'/f'rank{r}.json').read_text()) for r in range(8)]
    create_only = results[0]['options']['suite'] == 'create'
    minimal = bool(results[0]['options'].get('minimal', False))
    if minimal and not create_only: raise ValueError('minimal requires create suite')
    marker = 'EP8_EID_MINIMAL_RANK_PASS' if minimal else 'EP8_EID_CREATE_RANK_PASS' if create_only else 'EP8_EID_RANK_PASS'
    ranks = set(map(int, re.findall(r'^'+marker+r' rank=(\d+)$', (root/'run.log').read_text(), re.M)))
    if ranks != set(range(8)): raise ValueError('missing clean rank completion')
    for r, result in enumerate(results):
        if result['options']['suite'] != results[0]['options']['suite']: raise ValueError('different rank suite')
        if bool(result['options'].get('minimal', False)) != minimal: raise ValueError('different minimal mode')
        if result['rank'] != r or result['backend'] != 'participant_eid_ccu_urma': raise ValueError('wrong result identity')
        for a, b in zip(result['reports'], results[0]['reports']):
            if any(a[k] != b[k] for k in ('case', 'counts', 'row_bytes', 'relay_rows', 'median_us')):
                raise ValueError('rank reports disagree')
        if len(result['reports']) != len(results[0]['reports']): raise ValueError('case count differs')
    return results[0], json.loads((root/'plans'/'topology.json').read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--compare', type=Path, help='native baseline result directory')
    args = p.parse_args()
    result, topology = load(args.run)
    if result['options']['suite'] == 'create':
        if args.compare: raise ValueError('create-only cannot be used as a performance result')
        if result['options'].get('minimal', False):
            print('EP8_EID_MINIMAL_ALL_RANKS_AND_EXIT_PASS (configuration only; no EID channels or registration)')
        else:
            print('EP8_EID_CREATE_ALL_RANKS_AND_EXIT_PASS (registration only; no data validation)')
        return
    other = None
    if args.compare:
        other, other_topology = load(args.compare)
        for key in ('dtype', 'width', 'row_scale', 'warmup', 'iters', 'rounds', 'dynamic_iters'):
            if result['options'][key] != other['options'][key]: raise ValueError('different measurement option: '+key)
        for key in ('devices', 'allowed_relays', 'links', 'relay_routes'):
            if topology[key] != other_topology[key]: raise ValueError('different topology: '+key)
        if len(result['reports']) != len(other['reports']): raise ValueError('different case lists')
    for i, report in enumerate(result['reports']):
        baseline = other['reports'][i] if other else report
        for key in ('case', 'counts', 'row_bytes'):
            if report[key] != baseline[key]: raise ValueError('different payload: '+key)
        print(f"case={report['case']} row_bytes={report['row_bytes']} executed_relay={report['executed_relay']}")
        for metric, values in report['median_us'].items():
            comparison = baseline['median_us'][metric]
            print(metric + ': ' + ' '.join(f'{mode}={us:.1f}us' for mode, us in values.items()))
            if args.compare and ('native' not in comparison or 'multipath' not in values): raise ValueError('compare needs native baseline and multipath candidate')
            if 'native' in comparison and 'multipath' in values:
                ratio = comparison['native']/values['multipath']
                print(f'  native/multipath={ratio:.3f} (>1 means multipath faster)')
    print('EP8_EID_ALL_RANKS_AND_EXIT_PASS')


if __name__ == '__main__': main()
