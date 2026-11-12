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
    probe = results[0]['options'].get('register_stage', 'full') != 'full'
    if probe and (not create_only or minimal): raise ValueError('invalid registration probe suite')
    marker = 'EP8_EID_REGISTER_PROBE_RANK_PASS' if probe else 'EP8_EID_MINIMAL_RANK_PASS' if minimal else 'EP8_EID_CREATE_RANK_PASS' if create_only else 'EP8_EID_RANK_PASS'
    # Old print() workers can concatenate markers or append SDK output. New
    # workers use one atomic write; retain compatibility with existing logs,
    # including rank=7[INFO] with no separator. Consume the full rank number.
    ranks = set(map(int, re.findall(re.escape(marker)+r' rank=(\d+)(?=\s|\[|EP8_|$)', (root/'run.log').read_text())))
    if ranks != set(range(8)):
        raise ValueError(f'missing clean rank completion: expected_marker={marker} found={sorted(ranks)} '
                         f'minimal={minimal} probe={probe} script={Path(__file__).resolve()}')
    for r, result in enumerate(results):
        if result['options']['suite'] != results[0]['options']['suite']: raise ValueError('different rank suite')
        if bool(result['options'].get('minimal', False)) != minimal: raise ValueError('different minimal mode')
        if result['options'].get('register_stage', 'full') != results[0]['options'].get('register_stage', 'full'): raise ValueError('different registration stages')
        if result['rank'] != r or result['backend'] != 'participant_eid_ccu_urma': raise ValueError('wrong result identity')
        for a, b in zip(result['reports'], results[0]['reports']):
            if any(a[k] != b[k] for k in ('case', 'counts', 'row_bytes', 'relay_rows', 'median_us')):
                raise ValueError('rank reports disagree')
        if len(result['reports']) != len(results[0]['reports']): raise ValueError('case count differs')
    return results[0], json.loads((root/'plans'/'topology.json').read_text())


def traffic_fields(plan):
    # relay_bytes is payload assigned to any alternate path, NOT extra traffic
    # for L1. Keep old report dictionaries readable without guessing extra bytes.
    original = plan.get('original_network_bytes')
    transmitted = plan.get('transmitted_network_bytes')
    extra = 'unknown' if original is None or transmitted is None else transmitted - original
    if extra != 'unknown' and extra < 0:
        raise ValueError('transmitted network bytes smaller than original payload')
    return (f"moved_bytes={plan['relay_bytes']} extra_network_bytes={extra} "
            f"l1_rows={plan.get('l1_rows', 0)} "
            f"participant_relay_rows={plan.get('participant_relay_rows', 'unknown')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--compare', type=Path, help='native baseline result directory')
    args = p.parse_args()
    result, topology = load(args.run)
    if result['options']['suite'] == 'create':
        if args.compare: raise ValueError('create-only cannot be used as a performance result')
        if result['options'].get('register_stage', 'full') != 'full':
            print('EP8_EID_REGISTER_PROBE_ALL_RANKS_AND_EXIT_PASS stage=' + result['options']['register_stage'] + ' (registration probe only; payload disabled)')
        elif result['options'].get('minimal', False):
            print('EP8_EID_MINIMAL_ALL_RANKS_AND_EXIT_PASS (configuration only; no EID channels or registration)')
        else:
            print('EP8_EID_CREATE_ALL_RANKS_AND_EXIT_PASS (registration only; no data validation)')
        return
    other = None
    if args.compare:
        other, other_topology = load(args.compare)
        if other['options']['suite'] == 'create': raise ValueError('create/probe cannot be a performance baseline')
        for key in ('dtype', 'width', 'row_scale', 'warmup', 'iters', 'rounds', 'dynamic_iters', 'planner', 'counts_path', 'combination_audit', 'dynamic_stage_audit'):
            if result['options'].get(key, 'greedy' if key == 'planner' else 'legacy' if key == 'counts_path' else False if key in ('combination_audit', 'dynamic_stage_audit') else None) != other['options'].get(key, 'greedy' if key == 'planner' else 'legacy' if key == 'counts_path' else False if key in ('combination_audit', 'dynamic_stage_audit') else None): raise ValueError('different measurement option: '+key)
        for key in ('devices', 'allowed_relays', 'links', 'relay_routes'):
            if topology[key] != other_topology[key]: raise ValueError('different topology: '+key)
        if len(result['reports']) != len(other['reports']): raise ValueError('different case lists')
    for i, report in enumerate(result['reports']):
        baseline = other['reports'][i] if other else report
        for key in ('case', 'counts', 'row_bytes'):
            if report[key] != baseline[key]: raise ValueError('different payload: '+key)
        print(f"case={report['case']} row_bytes={report['row_bytes']} executed_relay={report['executed_relay']}")
        print('counts_path=' + result['options'].get('counts_path', 'legacy'))
        if 'plan' in report:
            plan = report['plan']
            print(f"planner={report.get('planner')} actions={len(plan['actions'])} "
                  f"relay_rows={report['relay_rows']} evaluations={plan['candidate_evaluations']} "
                  f"peak_bytes={plan['direct_peak_bytes']}->{plan['planned_link_peak_bytes']} "
                  f"{traffic_fields(plan)} stop={plan['stop_reason']}")
        for metric, values in report['median_us'].items():
            if metric not in baseline['median_us']: raise ValueError('baseline missing timing metric: '+metric)
            comparison = baseline['median_us'][metric]
            print(metric + ': ' + ' '.join(f'{mode}={us:.1f}us' for mode, us in values.items()))
            if args.compare and ('native' not in comparison or 'multipath' not in values): raise ValueError('compare needs native baseline and multipath candidate')
            if 'native' in comparison and 'multipath' in values:
                ratio = comparison['native']/values['multipath']
                print(f'  native/multipath={ratio:.3f} (>1 means multipath faster)')
    print('EP8_EID_ALL_RANKS_AND_EXIT_PASS')


if __name__ == '__main__': main()
