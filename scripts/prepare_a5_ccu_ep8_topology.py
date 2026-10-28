#!/usr/bin/env python3
"""Generate EP8 direct endpoint bindings and participant-relay candidates.

All identities come from rootinfo + Atlas + live hccn, never a handwritten TXT.
The generated direct manifest is checked against HCCL's actual selected EIDs at
runtime. Conservative first version: one verified link per pair, one local die
per rank, distinct local ports per peer. No guessed bandwidth weights.
"""
import argparse
import csv
import json
from pathlib import Path

from a5_ccu_machine_topology import add_machine_arguments, load_machine, directed_links, connected_pair
from prepare_a5_ccu_peer_plan import FIELDS, parse_phys_ids


def build_topology(devices, entries, edges, net_layer=0, direct_route=0):
    if len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError('exactly eight distinct participant devices are required')
    rows, links = [], {}
    dies, ports = {r: set() for r in range(8)}, {r: set() for r in range(8)}
    for s in range(8):
        for d in range(s + 1, 8):
            matches = {}
            for link in directed_links(edges, devices[s], devices[d], net_layer):
                for left, right in connected_pair(entries, devices[s], link['endpoint_port'],
                                                  devices[d], link['relay_port'], net_layer):
                    identity = (left['eid'], right['eid'], tuple(link['endpoint_port']), tuple(link['relay_port']))
                    matches[identity] = (left, right, link)
            if len(matches) != 1:
                raise ValueError(f'pair {devices[s]}->{devices[d]} needs exactly one verified UB_CTP link; '
                                 f'found {len(matches)}; ambiguous/missing links are not guessed')
            left, right, link = next(iter(matches.values()))
            for rank, item in ((s, left), (d, right)):
                port = (item['die'], item['port'])
                if port in ports[rank]:
                    raise ValueError(f'rank {rank}: shared physical port; independent-link model unsupported')
                ports[rank].add(port)
                dies[rank].add(item['die'])
            rows.append(dict(src_rank=s, dst_rank=d, src_phy=devices[s], dst_phy=devices[d],
                             kind='direct', relay_phy='-', direct_route=direct_route, weight=2,
                             src_die=left['die'], dst_die=right['die'],
                             src_eid=left['eid'], dst_eid=right['eid']))
            links[f'{s},{d}'] = dict(src_port=[left['die'], left['port']],
                                    dst_port=[right['die'], right['port']], edge=link['edge'],
                                    src_eid=left['eid'], dst_eid=right['eid'])
    if any(len(value) != 1 for value in dies.values()):
        raise ValueError('EP8 direct CCU executor requires one local IO die per rank')
    # All eight ranks are active endpoints. Both legs terminate in participant
    # HBM; no external relay or synthetic endpoint address is constructed.
    allowed = [[k for k in range(8) if k not in (s, d)] if s != d else []
               for s in range(8) for d in range(8)]
    topology = dict(version=1, executor='participant_hbm_two_stage_ccu', devices=devices,
                    model='equal_bandwidth_independent_directed_links',
                    allowed_relays=allowed, links=links,
                    rank_dies=[next(iter(dies[r])) for r in range(8)])
    return rows, topology


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    p.add_argument('--direct-route', type=int, default=0)
    p.add_argument('--output-dir', type=Path, required=True)
    add_machine_arguments(p)
    args = p.parse_args()
    try:
        if args.direct_route < 0:
            raise ValueError('direct-route must be nonnegative')
        devices = parse_phys_ids(args.devices, '--devices')
        entries, edges, provenance = load_machine(args, devices)
        rows, topology = build_topology(devices, entries, edges, args.net_layer, args.direct_route)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'peer_plan.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    topology['provenance'] = provenance
    (args.output_dir / 'topology.json').write_text(json.dumps(topology, indent=2) + '\n')
    print('EP8_TOPOLOGY_PASS ' + str(args.output_dir))


if __name__ == '__main__':
    main()
