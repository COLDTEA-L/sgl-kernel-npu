#!/usr/bin/env python3
"""Versioned controller boundary for single-port L1 CCU paths. No EID synthesis."""
import hashlib
import json
from pathlib import Path
from a5_ccu_machine_topology import normalize_eid, port_list


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def single_port(eid):
    raw = bytes.fromhex(normalize_eid(eid).replace(':',''))
    die, port = raw[5] >> 6, raw[5] & 63
    if die > 1 or port > 8:
        raise ValueError('not an Ascend950 single-port EID (aggregate/unsupported port)')
    return die, port


def memberships(root):
    out = {}
    if root.get('version') != '2.0':
        raise ValueError('rootinfo version 2.0 required')
    for rank in root['rank_list']:
        phy = int(rank['device_id'])
        for level in rank['level_list']:
            if int(level['net_layer']) != 1:
                continue
            if level['net_type'] != 'CLOS':
                raise ValueError('L1 layer must declare CLOS')
            for address in level['rank_addr_list']:
                if address['addr_type'].upper() != 'EID':
                    continue
                for die, port in port_list(address['ports']):
                    key = (phy, die, port)
                    value = (str(level['net_instance_id']), str(address['plane_id']))
                    if key in out and out[key] != value:
                        raise ValueError('ambiguous L1 physical port membership')
                    out[key] = value
    return out


def network_fes(root):
    """Keep the FE namespace declared by each L1 rootinfo endpoint group."""
    out={}
    for rank in root['rank_list']:
        for level in rank['level_list']:
            if int(level['net_layer'])!=1: continue
            for a in level['rank_addr_list']:
                if a['addr_type'].upper()!='EID': continue
                raw=bytes.fromhex(normalize_eid(a['addr']).replace(':',''))
                for die,port in port_list(a['ports']):
                    out.setdefault((int(rank['device_id']),die,port),set()).add(raw[6]&127)
    return out


def validate_config(path, devices, root_path=None):
    path = Path(path)
    c = json.loads(path.read_text())
    if c.get('schema') != 'a5.single-port-l1.v1' or not isinstance(c.get('generation'), str) or not c['generation']:
        raise ValueError('requires a5.single-port-l1.v1 and nonempty generation')
    if c.get('devices') != list(devices):
        raise ValueError('L1 config device order differs from communicator')
    root_path = Path(root_path or c['rootinfo'])
    if sha(root_path) != c['rootinfo_sha256']:
        raise ValueError('rootinfo changed: rediscover L1 inventory and regenerate policy')
    invpath = path.parent / c['inventory']
    if sha(invpath) != c['inventory_sha256']:
        raise ValueError('L1 inventory digest mismatch')
    inv = json.loads(invpath.read_text())
    if inv.get('schema') != 'a5.endpoint-inventory.v1' or inv.get('rootinfo_sha256') != c['rootinfo_sha256']:
        raise ValueError('inventory is not bound to this rootinfo')
    root=json.loads(root_path.read_text())
    members = memberships(root)
    fes = network_fes(root)
    observed = {(int(e['device_id']), normalize_eid(e['eid']))
                for e in inv['endpoints'] if e['protocol'] == 4 and e['addr_type'] in (1, 3)}
    pairs, ids = set(), set()
    for p in c['paths']:
        a, b = p['src_phy'], p['dst_phy']
        if a not in devices or b not in devices or devices.index(a) >= devices.index(b):
            raise ValueError('L1 path needs a canonical communicator pair')
        if (a, b) in pairs or not p['id'] or p['id'] in ids:
            raise ValueError('one L1 path per pair and unique controller path IDs required')
        pairs.add((a, b)); ids.add(p['id'])
        for side, phy in (('src', a), ('dst', b)):
            eid = normalize_eid(p[side+'_eid'])
            if (phy, eid) not in observed:
                raise ValueError('EID absent from device-local inventory; never synthesize addresses')
            die, port = single_port(eid)
            if (die, port) != (p[side+'_die'], p[side+'_port']):
                raise ValueError('EID classification disagrees with explicit port/die')
            if bytes.fromhex(eid.replace(':',''))[6]&127 not in fes.get((phy,die,port),set()):
                raise ValueError('EID FE differs from L1 rootinfo network namespace')
            if members.get((phy, die, port)) != (p['net_instance_id'], p['plane_id']):
                raise ValueError('endpoints must belong to the same declared L1 plane/instance')
        for key in ('l0_weight', 'l1_weight'):
            if type(p[key]) is not int or not 0 < p[key] <= 65535:
                raise ValueError('path weights must be positive bounded integers')
    if not pairs:
        raise ValueError('empty L1 policy')
    return c


def apply_l1(rows, devices, path, root_path=None):
    c = validate_config(path, devices, root_path)
    rows = [dict(r) for r in rows]
    resources = [[] for _ in range(len(devices)**2)]
    for p in c['paths']:
        s, d = devices.index(p['src_phy']), devices.index(p['dst_phy'])
        direct = next(r for r in rows if r['src_rank']==s and r['dst_rank']==d and r['kind']=='direct')
        # Bound L0 descriptors must be produced using the current machine topology.
        if direct['src_eid']=='-' or direct['dst_eid']=='-':
            raise ValueError('L1 requires machine-bound L0 direct rows')
        if (direct['src_die'],direct['dst_die']) != (p['src_die'],p['dst_die']):
            raise ValueError('cross-die L0+L1 scheduling is not supported')
        if single_port(direct['src_eid']) == (p['src_die'],p['src_port']) or single_port(direct['dst_eid']) == (p['dst_die'],p['dst_port']):
            raise ValueError('L1 must not alias the L0 physical port')
        direct['weight']=p['l0_weight']
        rows.append(dict(direct,kind='l1_single',relay_phy='-',weight=p['l1_weight'],
                         src_eid=normalize_eid(p['src_eid']),dst_eid=normalize_eid(p['dst_eid'])))
        a=s*18+p['src_die']*9+p['src_port']; b=d*18+p['dst_die']*9+p['dst_port']
        resources[s*len(devices)+d]=[a,b]; resources[d*len(devices)+s]=[b,a]
    rows.sort(key=lambda r:(r['src_rank'],r['dst_rank'],{'direct':0,'relay':1,'l1_single':2}[r['kind']]))
    return rows, dict(schema=c['schema'],generation=c['generation'],config_sha256=sha(path),
                      inventory_sha256=c['inventory_sha256'],rootinfo_sha256=c['rootinfo_sha256'],
                      paths=c['paths'],l1_resources=resources,
                      physical_single_port_verified=False)
