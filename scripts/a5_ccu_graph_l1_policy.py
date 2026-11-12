#!/usr/bin/env python3
"""Compile declared L1 candidates; route must match each to the live rank graph.

No EID synthesis, DCMI classification, rootinfo mutation or optimistic per-peer
bandwidth. One declared endpoint group per rank, charged as ONE shared resource
at a conservative one-L0-link capacity (even if it contains multiple ports).
"""
import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

from a5_ccu_machine_topology import normalize_eid, port_list
from a5_ccu_l1_policy import sha

SCHEMA = 'a5.rank-graph-l1.v1'


def declared_endpoints(root, devices):
    if root.get('version') != '2.0':
        raise ValueError('rootinfo version 2.0 required')
    out = {d: [] for d in devices}
    for rank in root['rank_list']:
        phy = int(rank['device_id'])
        if phy not in out:
            continue
        for level in rank['level_list']:
            if int(level['net_layer']) != 1:
                continue
            if level['net_type'] != 'CLOS':
                raise ValueError('graph L1 policy requires declared CLOS layer 1')
            for address in level['rank_addr_list']:
                if address['addr_type'].upper() != 'EID':
                    continue
                ports = sorted(port_list(address['ports']))
                if not ports or len(set(ports)) != len(ports) or any(d not in (0, 1) or not 0 <= p <= 8 for d, p in ports):
                    raise ValueError('invalid or duplicate physical L1 ports')
                # Current CCU executor cannot schedule an endpoint across dies.
                if len({d for d, _ in ports}) != 1:
                    continue
                eid = normalize_eid(address['addr'])
                if not any(bytes.fromhex(eid.replace(':', ''))):
                    raise ValueError('zero L1 EID')
                out[phy].append(dict(eid=eid, die=ports[0][0], ports=[p for _, p in ports],
                                     plane_id=str(address.get('plane_id', '0')),
                                     net_instance_id=str(level['net_instance_id'])))
    # Duplicate/overlapping aliases must not be counted as extra capacity.
    for endpoints in out.values():
        used = set()
        for e in endpoints:
            keys = {(e['die'], p) for p in e['ports']}
            if used & keys:
                raise ValueError('overlapping L1 endpoint declarations; do not mix aggregate and member EIDs')
            used |= keys
    return out


def generate(root_path, base_topology_path, output_path, plane=None):
    root_path, base_topology_path, output_path = map(Path, (root_path, base_topology_path, output_path))
    base = json.loads(base_topology_path.read_text())
    devices, dies = base['devices'], base['rank_dies']
    if len(devices) != 8 or len(set(devices)) != 8 or len(dies) != 8 or base.get('l1_resources') is not None:
        raise ValueError('requires a generated L0-only EP8 topology with rank_dies')
    if base.get('provenance', {}).get('root_info_sha256') != sha(root_path):
        raise ValueError('base topology must be generated from this exact rootinfo')
    candidates = declared_endpoints(json.loads(root_path.read_text()), devices)
    by_rank = [[e for e in candidates[d] if e['die'] == dies[r] and (plane is None or e['plane_id'] == plane)]
               for r, d in enumerate(devices)]
    common = None
    for eps in by_rank:
        keys = {(e['net_instance_id'], e['plane_id']) for e in eps}
        common = keys if common is None else common & keys
    if not common:
        raise ValueError('no common declared L1 plane on the L0 IO dies; no cross-plane fallback')
    choices = []
    for key in sorted(common):
        eps = [[e for e in entries if (e['net_instance_id'], e['plane_id']) == key] for entries in by_rank]
        # Ambiguity should be resolved by an explicit controller policy, not EID prefix guesses.
        if any(len(x) != 1 for x in eps):
            continue
        picked = [x[0] for x in eps]
        choices.append((sum(len(e['ports']) for e in picked), key, picked))
    if not choices:
        raise ValueError('ambiguous endpoints within declared plane')
    _, key, picked = min(choices, key=lambda x: (x[0], x[1]))
    paths = []
    for s, d in itertools.combinations(range(8), 2):
        a, b = picked[s], picked[d]
        paths.append(dict(id=f'graph-l1-{devices[s]}-{devices[d]}', src_phy=devices[s], dst_phy=devices[d],
                          src_eid=a['eid'], dst_eid=b['eid'], src_die=a['die'], dst_die=b['die'],
                          src_ports=a['ports'], dst_ports=b['ports'],
                          plane_id=key[1], net_instance_id=key[0], l0_weight=1, l1_weight=1))
    c = dict(schema=SCHEMA, generation=datetime.now(timezone.utc).isoformat(), devices=devices,
             rootinfo=str(root_path.resolve()), rootinfo_sha256=sha(root_path), paths=paths,
             selection='rootinfo_candidates_require_live_rank_graph_match',
             resource_model='shared_endpoint_group_conservative_1x')
    # Validate before creating output; no files in /etc or driver topology are changed.
    validate(c, devices, root_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(c, indent=2) + '\n')
    print(f'L1_GRAPH_POLICY_CANDIDATES {output_path} pairs={len(paths)} plane={key[1]} '
          'runtime_graph_match_required=1 group_capacity=1x')
    return c


def validate(c, devices, root_path):
    if (c.get('schema') != SCHEMA or c.get('devices') != list(devices) or len(devices) != 8
            or len(set(devices)) != 8 or any(type(d) is not int for d in devices)):
        raise ValueError('rank-graph L1 policy requires matching EP8 device order')
    if not isinstance(c.get('generation'), str) or not c['generation'] or c['rootinfo_sha256'] != sha(root_path):
        raise ValueError('missing generation or stale rootinfo')
    if c.get('resource_model') != 'shared_endpoint_group_conservative_1x' or 'resource_budget' in c:
        raise ValueError('graph L1 first version requires conservative shared-group budget')
    declared = declared_endpoints(json.loads(Path(root_path).read_text()), devices)
    seen, ids, groups = set(), set(), {}
    for p in c['paths']:
        a, b = p['src_phy'], p['dst_phy']
        if a not in devices or b not in devices or devices.index(a) >= devices.index(b) or (a, b) in seen:
            raise ValueError('invalid/duplicate canonical peer pair')
        seen.add((a, b))
        if not p['id'] or p['id'] in ids:
            raise ValueError('unique nonempty path id required')
        ids.add(p['id'])
        for side, phy in (('src', a), ('dst', b)):
            ports = p[side+'_ports']
            if (type(p[side+'_die']) is not int or not isinstance(ports, list) or
                    any(type(v) is not int for v in ports)):
                raise ValueError('die/ports must be exact integers')
            expected = dict(eid=normalize_eid(p[side+'_eid']), die=p[side+'_die'], ports=ports,
                            plane_id=p['plane_id'], net_instance_id=p['net_instance_id'])
            if expected not in declared[phy]:
                raise ValueError('endpoint is not an exact rootinfo L1 declaration')
            if phy in groups and groups[phy] != expected:
                raise ValueError('graph mode requires one shared L1 endpoint group per rank')
            groups[phy] = expected
        if type(p['l0_weight']) is not int or type(p['l1_weight']) is not int or p['l0_weight'] != 1 or p['l1_weight'] != 1:
            raise ValueError('initial graph L1 policy uses conservative 1:1 weights')
    if not seen:
        raise ValueError('empty graph L1 policy')
    return groups


def apply_graph_l1(rows, devices, path, root_path=None):
    path = Path(path)
    c = json.loads(path.read_text())
    groups = validate(c, devices, root_path or c['rootinfo'])
    rows = [dict(r) for r in rows]
    # Reuse the planner's 144 resource slots: min physical port is the ID of the
    # ENTIRE group, NOT one independently charged slot per peer or member port.
    resources = [[] for _ in range(64)]
    for p in c['paths']:
        s, d = devices.index(p['src_phy']), devices.index(p['dst_phy'])
        direct = next(r for r in rows if r['kind'] == 'direct' and r['src_rank'] == s and r['dst_rank'] == d)
        if (direct['src_die'], direct['dst_die']) != (p['src_die'], p['dst_die']):
            raise ValueError('L1 graph endpoint must use the L0 IO die')
        from a5_ccu_l1_policy import single_port
        for side in ('src', 'dst'):
            die, port = single_port(direct[side+'_eid'])
            if die == p[side+'_die'] and port in p[side+'_ports']:
                raise ValueError('L1 group overlaps L0 direct port')
        direct['weight'] = 1
        rows.append(dict(direct, kind='l1_graph', relay_phy='-', weight=1,
                         src_eid=normalize_eid(p['src_eid']), dst_eid=normalize_eid(p['dst_eid'])))
        a = s*18 + p['src_die']*9 + min(p['src_ports'])
        b = d*18 + p['dst_die']*9 + min(p['dst_ports'])
        resources[s*8+d], resources[d*8+s] = [a, b], [b, a]
    budget = dict(schema='a5.l1-resource-budget.v1', generation=c['generation'],
                  tx_capacity_permille=[1000]*144, rx_capacity_permille=[1000]*144,
                  tx_background_bytes=[0]*144, rx_background_bytes=[0]*144)
    rows.sort(key=lambda r: (r['src_rank'], r['dst_rank'], {'direct':0, 'relay':1, 'l1_graph':2}[r['kind']]))
    return rows, dict(schema=SCHEMA, generation=c['generation'], config_sha256=sha(path),
                      rootinfo_sha256=c['rootinfo_sha256'], paths=c['paths'], l1_resources=resources,
                      l1_budget=budget, endpoint_mode='rank_graph', runtime_graph_match_required=True,
                      resource_model=c['resource_model'], resource_groups=groups,
                      physical_single_port_verified=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root-info', type=Path, required=True)
    p.add_argument('--base-topology', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plane', help='declared plane_id; default prefers the smallest common same-die group')
    a = p.parse_args()
    try:
        generate(a.root_info, a.base_topology, a.output, a.plane)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
