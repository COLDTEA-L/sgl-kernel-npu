#!/usr/bin/env python3
"""Prepare two explicit relay catalogs sharing a verified Atlas physical link.

No NPU task is launched. Uses current rootinfo + Atlas + read-only hccn queries.
"""
import argparse
import itertools
import json
from pathlib import Path
import shutil

from a5_ccu_machine_topology import add_machine_arguments, load_machine, relay_candidates
from a5_physical_resources import PhysicalInventory, row_links
from resolve_a5_explicit_multirelay_eids import FIELDS, csv_ints


def catalogs(entries, edges, devices, relays, layer, inventory):
    choices = []
    for relay in relays:
        valid = []
        for row in relay_candidates(entries, edges, *devices, relay, 'all', layer):
            row.update(src_phy=devices[0], dst_phy=devices[1], route_id=1000 + relay, weight=1)
            try:
                inventory.validate(row)
            except ValueError:
                continue
            valid.append(row)
        if not valid:
            raise ValueError(f'no uniquely port-bound live route for {devices} via {relay}')
        choices.append(valid)
    result = []
    for rows in itertools.product(*choices):
        if len({r['src_die'] for r in rows}) == 1 and len({r['dst_die'] for r in rows}) == 1:
            result.append(list(rows))
    if not result:
        raise ValueError('no relay catalog using one local IO die per endpoint')
    return result


def choose(a_catalogs, b_catalogs, shared):
    selected, best = None, -1
    for a in a_catalogs:
        ar = {link for row in a for link in row_links(row)}
        for b in b_catalogs:
            br = {link for row in b for link in row_links(row)}
            overlap = sorted(ar & br)
            if bool(overlap) != shared:
                continue
            if len(overlap) > best:
                selected, best = (a, b, overlap), len(overlap)
    if selected is None:
        raise ValueError('no requested physical-link overlap/disjoint plan exists for these card roles; change card IDs, not resource names')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_machine_arguments(parser)
    parser.add_argument('--managed-devices', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--scenario', choices=['shared', 'disjoint'], required=True)
    parser.add_argument('--a-devices', default='2,3')
    parser.add_argument('--a-relays', default='0,1')
    parser.add_argument('--b-devices', help='default shared=0,1; disjoint=4,5')
    parser.add_argument('--b-relays', help='default shared=2,3; disjoint=6,7')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        managed = csv_ints(args.managed_devices, '--managed-devices', require_unique=True)
        a = csv_ints(args.a_devices, '--a-devices', require_unique=True)
        b = csv_ints(args.b_devices or ('0,1' if args.scenario == 'shared' else '4,5'), '--b-devices', require_unique=True)
        ra = csv_ints(args.a_relays, '--a-relays', require_unique=True)
        rb = csv_ints(args.b_relays or ('2,3' if args.scenario == 'shared' else '6,7'), '--b-relays', require_unique=True)
        if len(managed) != 8 or len(a) != 2 or len(b) != 2 or set(a) & set(b):
            raise ValueError('need eight managed cards and disjoint pairs of worker endpoints')
        for endpoints, relays in [(a, ra), (b, rb)]:
            if len(relays) not in (1, 2) or set(endpoints) & set(relays) or not set(endpoints + relays) <= set(managed):
                raise ValueError('invalid endpoint/relay card roles')
        out = args.output_dir.resolve()
        out.mkdir(parents=True, exist_ok=True)
        if any((out / name).exists() for name in ('config.json', 'topology.json', 'provenance.json', 'a.tsv', 'b.tsv')):
            raise ValueError('output plan files already exist; choose a fresh output directory')
        args.diagnostics_json = out / 'provenance.json'
        entries, edges, provenance = load_machine(args, set(managed))
        shutil.copyfile(provenance['topology_json'], out / 'topology.json')
        inventory = PhysicalInventory(out / 'topology.json', out / 'provenance.json')
        ac = catalogs(entries, edges, a, ra, args.net_layer, inventory)
        bc = catalogs(entries, edges, b, rb, args.net_layer, inventory)
        ar, br, overlap = choose(ac, bc, args.scenario == 'shared')
        for name, rows in [('a', ar), ('b', br)]:
            (out / (name + '.tsv')).write_text('\t'.join(FIELDS) + '\n' +
                ''.join('\t'.join(str(row[field]) for field in FIELDS) + '\n' for row in rows))
        config = dict(resource_mode='physical_link', expectation=args.scenario, managed_devices=managed,
                      topology_json='topology.json', provenance_json='provenance.json', jobs=[
                          dict(name='a', devices=a, relay_phys=ra, relay_manifest='a.tsv', policy='mixed'),
                          dict(name='b', devices=b, relay_phys=rb, relay_manifest='b.tsv', policy='mixed')])
        (out / 'config.json').write_text(json.dumps(config, indent=2))
        (out / 'selection.json').write_text(json.dumps(dict(shared_payload_links=overlap,
            a_paths=[row_links(r) for r in ar], b_paths=[row_links(r) for r in br],
            candidate_catalogs=[len(ac), len(bc)]), indent=2))
        print('CONFIG=' + str(out / 'config.json'))
        print(json.dumps(dict(shared_payload_links=overlap), indent=2))
        if args.scenario == 'shared' and not overlap:
            raise ValueError('internal error: empty physical conflict')
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
