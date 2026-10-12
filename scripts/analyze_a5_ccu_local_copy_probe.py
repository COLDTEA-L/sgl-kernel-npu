#!/usr/bin/env python3
"""Fail-closed proof summary for the isolated hardware CCU local-copy test."""
import argparse
import csv
import json
from pathlib import Path
import re


def analyze(root):
    settings = json.loads((root / 'run_settings.json').read_text())
    ranks = len(settings['devices'])
    rows = []
    for case in settings['cases']:
        directory = root / 'cases' / case['name']
        checked, reasons = [], []
        for rank in range(ranks):
            try:
                status = int((directory / f'rank{rank}.status').read_text())
                log = (directory / f'rank{rank}.log').read_text()
                events = [json.loads(line.split(' ', 1)[1])
                          for line in log.splitlines() if line.startswith('LOCAL_COPY_RESULT_JSON ')]
                marker = re.search(rf'^LOCAL_COPY_WORKER_PASS rank={rank} ', log, re.MULTILINE)
                valid = (len(events) == 1 and isinstance(events[0], dict) and
                    all(type(events[0].get(key)) is int for key in
                        ('rank', 'bytes', 'changed_calls', 'queued_calls', 'channel_count')) and
                    events[0].get('acl_d2d') is False and events[0] == dict(rank=rank,
                    copy_mode=case['copy_mode'], bytes=case['bytes'],
                    changed_calls=settings['changed_calls'], queued_calls=settings['queued_calls'],
                    guards='PASS', status='PASS', engine='CCU', channel_count=0, acl_d2d=False))
                if status != 0 or not marker or not valid or 'LOCAL_COPY_FAILURE' in log:
                    raise ValueError(f'status={status}, result/cleanup evidence incomplete')
                checked.append(rank)
            except (OSError, ValueError, TypeError) as error:
                reasons.append(f'rank{rank}: {error}')
        try:
            process_status = int((directory / 'status.txt').read_text())
        except (OSError, ValueError):
            process_status = 'MISSING'
        passed = (not settings.get('builder_only', False) and process_status == 0 and
                  checked == list(range(ranks)) and not reasons)
        rows.append(dict(case=case['name'], copy_mode=case['copy_mode'], bytes=case['bytes'],
            status=process_status, result='PASS' if passed else 'FAIL_OR_INCOMPLETE',
            checked_ranks=','.join(map(str, checked)), reason='; '.join(reasons)))
    complete = bool(rows) and not settings.get('builder_only', False) and all(row['result'] == 'PASS' for row in rows)
    summary = dict(complete=complete, conclusion='PASS' if complete else 'FAIL_OR_INCOMPLETE',
        hardware_data_pass_cases=[row['case'] for row in rows if row['result'] == 'PASS'], cases=rows)
    with (root / 'local_copy_summary.tsv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=('case', 'copy_mode', 'bytes', 'status', 'result', 'checked_ranks', 'reason'), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    (root / 'local_copy_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = ['# Isolated CCU LocalCopyNb + WaitEvent proof', '',
        f'- conclusion: **{summary["conclusion"]}**', f'- physical devices: `{settings["devices"]}`',
        '- No remote WriteNb, data ChannelAcquire, or ACL D2D copy.',
        '- HCCL communicator initialization may use control-plane communication.',
        '- H2D input/poison and D2H snapshots are verification, not local-copy substitutes.', '',
        '| case | bytes | process status | data + guards + cleanup |', '|---|---:|---:|---|']
    for row in rows:
        report.append(f'| {row["case"]} | {row["bytes"]} | {row["status"]} | {row["result"]} |')
    report += ['', 'PASS requires every requested rank/case to complete device work, verify every queued snapshot,',
        'preserve guards/input, and finish cleanup with status=0. Builder-only is never hardware proof.',
        'A direct-large failure does not invalidate a smaller or chunked data PASS.',
        'A registration/context failure is not proof that hardware cannot copy locally.',
        'These timings are not a bandwidth benchmark; mixed H2D/D2H verification is intentional.', '', '## Diagnostics']
    report += [f'- {row["case"]}: {row["reason"]}' for row in rows if row['reason']]
    (root / 'local_copy_report.md').write_text('\n'.join(report) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    print(args.run_dir / 'local_copy_report.md')
    for row in result['cases']:
        print(f'{row["case"]}: {row["result"]} (process status={row["status"]})')
    if result['complete']:
        label = ('GroupCopy8x32K (HBM-MS-HBM)' if all(row['copy_mode'] == 'group' for row in result['cases'])
                 else 'LocalCopyNb + WaitEvent')
        print(f'PASS: {label} completed on device; changed payload, every queued snapshot, guards and cleanup verified.')
    else:
        print('FAIL_OR_INCOMPLETE: no overall hardware PASS; inspect per-case evidence and rank logs.')
    raise SystemExit(0 if result['complete'] else 1)


if __name__ == '__main__':
    main()
