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


def choose_single_port_pair(pairs, assignments):
    """Static catalog spreading; counts physical ports, not EID aliases.

    This is not per-call routing or a promise that network paths are independent.
    Controller-supplied policies can override this deterministic default.
    """
    def resource(endpoint):
        return endpoint['device_id'], endpoint['die'], endpoint['port']
    def score(pair):
        a, b = pair
        x, y = assignments.get(resource(a), 0), assignments.get(resource(b), 0)
        return max(x, y), x+y, resource(a), resource(b), a['eid'], b['eid']
    a, b = min(pairs, key=score)
    for e in (a, b):
        key = resource(e)
        assignments[key] = assignments.get(key, 0) + 1
    return a, b


def compile_budget(config, devices):
    """Map controller physical IDs into this EP8's local resource namespace."""
    budget = config.get('resource_budget')
    if budget is None:
        return None
    if len(devices) != 8 or not isinstance(budget, dict) or set(budget) != {'generation', 'ports'}:
        raise ValueError('resource_budget is EP8 only and requires generation/ports')
    if not isinstance(budget['generation'], str) or not budget['generation']:
        raise ValueError('resource budget generation required')
    out = dict(schema='a5.l1-resource-budget.v1', generation=budget['generation'],
               tx_capacity_permille=[1000]*144, rx_capacity_permille=[1000]*144,
               tx_background_bytes=[0]*144, rx_background_bytes=[0]*144)
    known = {(p[side+'_phy'], p[side+'_die'], p[side+'_port'])
             for p in config['paths'] for side in ('src', 'dst')}
    seen = set()
    if not isinstance(budget['ports'], list):
        raise ValueError('budget ports must be a list')
    keys = {'device_id', 'die', 'port', 'tx_capacity_permille', 'rx_capacity_permille',
            'tx_background_bytes', 'rx_background_bytes'}
    for item in budget['ports']:
        if not isinstance(item, dict) or set(item) != keys or any(type(v) is not int for v in item.values()):
            raise ValueError('budget port fields must be exact integers')
        key = (item['device_id'], item['die'], item['port'])
        if key not in known or key in seen:
            raise ValueError('budget must name a unique configured L1 physical port')
        seen.add(key)
        r = devices.index(key[0])*18 + key[1]*9 + key[2]
        for direction in ('tx', 'rx'):
            cap, base = item[direction+'_capacity_permille'], item[direction+'_background_bytes']
            if not 0 <= cap <= 1000 or not 0 <= base <= (1<<63)-1 or (cap == 0 and base):
                raise ValueError('invalid single-port capacity/background bytes')
            out[direction+'_capacity_permille'][r] = cap
            out[direction+'_background_bytes'][r] = base
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
    compile_budget(c, devices)
    return c


def apply_l1(rows, devices, path, root_path=None):
    if json.loads(Path(path).read_text()).get('schema') == 'a5.rank-graph-l1.v1':
        from a5_ccu_graph_l1_policy import apply_graph_l1
        return apply_graph_l1(rows, devices, path, root_path)
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
                      paths=c['paths'],l1_resources=resources,l1_budget=compile_budget(c,devices),
                      physical_single_port_verified=False)
