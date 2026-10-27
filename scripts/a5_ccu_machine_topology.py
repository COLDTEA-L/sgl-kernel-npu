#!/usr/bin/env python3
"""Join explicit rootinfo port bindings, Atlas edges and live hccn inventory.

No bits of an EID, UDMAC name, or EID index are interpreted as a port/device.
All subprocesses are read-only queries performed once during plan preparation.
"""
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import sys

ROOTINFO_PATHS = (Path('/etc/hccl_rootinfo.json'), Path('/etc/hccl_rootInfo.json'))
ATLAS_PATHS = tuple(Path('/usr/local/Ascend/driver/topo/950') / name
                    for name in ('atlas_950.json', 'atlas_950_1.json'))
HCCN_TOOL = '/usr/local/Ascend/driver/tools/hccn_tool'
PORT_RE = re.compile(r'^(\d+)/(\d+)$')


def integer(value, name):
    if isinstance(value, bool) or not re.fullmatch(r'\d+', str(value)):
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def parse_port(value):
    match = PORT_RE.fullmatch(str(value))
    if not match or int(match[1]) not in (0, 1):
        raise ValueError(f'invalid port {value!r}; expected DieID/PortID with die 0 or 1')
    return int(match[1]), int(match[2])


def port_list(values):
    if not isinstance(values, list) or not values:
        raise ValueError('ports must be a non-empty array of DieID/PortID strings')
    ports = [parse_port(value) for value in values]
    if len(set(ports)) != len(ports):
        raise ValueError('duplicate port in port list')
    return ports


def normalize_eid(value):
    if not isinstance(value, str):
        raise ValueError('EID must be a string')
    if re.fullmatch(r'[0-9a-fA-F]{32}', value):
        value = ':'.join(value[i:i + 4] for i in range(0, 32, 4))
    if '%' in value:
        raise ValueError('scoped IP address is not an EID')
    try:
        address = ipaddress.IPv6Address(value)
    except ipaddress.AddressValueError as exc:
        raise ValueError(f'invalid 128-bit EID {value!r}') from exc
    if int(address) == 0:
        raise ValueError('zero EID is not usable')
    return address.exploded


def select_json_path(explicit, candidates, option):
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise ValueError(f'{option}: file not found: {path}')
        return path
    found = [p for p in candidates if p.is_file()]
    if not found:
        raise ValueError(f'{option}: no current-machine file found; specify its path explicitly')
    contents = [json.loads(p.read_text()) for p in found]
    if any(value != contents[0] for value in contents[1:]):
        raise ValueError(f'{option}: conflicting automatic candidates {found}; select explicitly')
    return found[0]


def load_topology(path):
    data = json.loads(Path(path).read_text())
    if data.get('version') != '2.0':
        raise ValueError('Atlas topology version must be 2.0')
    peers = [integer(p['local_id'], 'Atlas physical local_id') for p in data['peer_list']]
    edges = data['edge_list']
    if len(set(peers)) != len(peers) or integer(data['peer_count'], 'peer_count') != len(peers):
        raise ValueError('topology peer_count/peer_list mismatch or duplicate physical ID')
    if integer(data['edge_count'], 'edge_count') != len(edges):
        raise ValueError('topology edge_count does not match edge_list')
    for edge in edges:
        integer(edge['net_layer'], 'net_layer')
        if not isinstance(edge['protocols'], list):
            raise ValueError('topology protocols must be a list')
        if edge.get('link_type') != 'PEER2PEER':
            continue
        a, b = (integer(edge[key], key) for key in ('local_a', 'local_b'))
        if a not in peers or b not in peers or a == b:
            raise ValueError('topology edge contains unknown or identical physical endpoints')
        pair_ports(edge['local_a_ports'], edge['local_b_ports'], f'{a}<->{b}')
    return set(peers), edges


def pair_ports(left, right, label):
    a, b = port_list(left), port_list(right)
    if len(a) != len(b):
        raise ValueError(f'{label}: unpaired topology ports')
    return list(zip(a, b))


def directed_links(edges, endpoint, relay, net_layer):
    links = []
    for ordinal, edge in enumerate(edges):
        if (int(edge['net_layer']) != net_layer or edge.get('link_type') != 'PEER2PEER'
                or 'UB_CTP' not in edge['protocols'] or edge.get('position', 'DEVICE') != 'DEVICE'):
            continue
        a, b = int(edge['local_a']), int(edge['local_b'])
        if (a, b) == (endpoint, relay):
            left, right = edge['local_a_ports'], edge['local_b_ports']
        elif (a, b) == (relay, endpoint):
            left, right = edge['local_b_ports'], edge['local_a_ports']
        else:
            continue
        for ep, rp in pair_ports(left, right, f'edge[{ordinal}]'):
            links.append(dict(edge=ordinal, endpoint_port=ep, relay_port=rp,
                              protocols=tuple(edge['protocols'])))
    return links


def required(mapping, key, location):
    if not isinstance(mapping, dict):
        raise ValueError(f'{location} must be an object')
    if key not in mapping:
        raise ValueError(f'missing required field {location}.{key}; '
                         f'available keys: {sorted(mapping)}')
    return mapping[key]


def root_bindings(data, devices, net_layer=0, net_instance_id=None, plane_id=None):
    if data.get('version') != '2.0' or data.get('status', 'completed') != 'completed':
        raise ValueError('rootinfo must have version 2.0 and completed status when status is present')
    ranks = required(data, 'rank_list', 'rootinfo')
    if not isinstance(ranks, list) or any(not isinstance(r, dict) for r in ranks):
        raise ValueError('rootinfo.rank_list must be an array of objects')
    if integer(required(data, 'rank_count', 'rootinfo'), 'rank_count') != len(ranks):
        raise ValueError('rootinfo rank_count does not match rank_list')
    # Local resource rootinfo need not be a collective rank table: rank_id
    # is not used to join physical devices/ports and must not be required.
    selected, instances = {}, set()
    for ordinal, rank in enumerate(ranks):
        rank_path = f'rootinfo.rank_list[{ordinal}]'
        device = integer(required(rank, 'device_id', rank_path), 'physical device_id')
        if device not in devices:
            continue
        local_id = integer(rank['local_id'], 'local_id') if 'local_id' in rank else None
        for level_index, level in enumerate(required(rank, 'level_list', rank_path)):
            level_path = f'{rank_path}.level_list[{level_index}]'
            if integer(required(level, 'net_layer', level_path), 'net_layer') != net_layer:
                continue
            instance = str(required(level, 'net_instance_id', level_path))
            if net_instance_id is not None and instance != net_instance_id:
                continue
            if device in selected:
                raise ValueError(f'duplicate/ambiguous device {device} at layer {net_layer}; '
                                 'select --net-instance-id for this machine')
            if required(level, 'net_type', level_path) != 'TOPO_FILE_DESC':
                raise ValueError('selected rootinfo layer must use TOPO_FILE_DESC')
            bindings = {}
            for address_index, address in enumerate(required(level, 'rank_addr_list', level_path)):
                address_path = f'{level_path}.rank_addr_list[{address_index}]'
                # Never reinterpret an IPv4/IPv6 host address as an EID.
                if required(address, 'addr_type', address_path).upper() != 'EID':
                    continue
                eid = normalize_eid(required(address, 'addr', address_path))
                plane = str(address.get('plane_id', '0'))
                for die, port in port_list(required(address, 'ports', address_path)):
                    key = (die, port)
                    item = dict(device=device, local_id=local_id, die=die, port=port,
                                eid=eid, plane_id=plane, net_layer=net_layer,
                                net_instance_id=instance)
                    if key in bindings and bindings[key] != item:
                        raise ValueError(f'ambiguous rootinfo binding: device {device} port {die}/{port}')
                    bindings[key] = item
            if plane_id is not None:
                bindings = {key: item for key, item in bindings.items()
                            if item['plane_id'] == plane_id}
            if not bindings:
                raise ValueError(f'device {device}: no explicit EID/ports bindings for selected layer/plane')
            selected[device] = bindings
            instances.add(instance)
    if set(selected) != set(devices):
        raise ValueError(f'rootinfo missing physical devices {sorted(set(devices) - set(selected))}')
    if len(instances) != 1:
        raise ValueError('requested devices belong to different net_instance_id values')
    local_ids = [next(iter(b.values()))['local_id'] for b in selected.values()
                 if next(iter(b.values()))['local_id'] is not None]
    if len(set(local_ids)) != len(local_ids):
        raise ValueError('duplicate rootinfo local_id within selected machine')
    return [item for bindings in selected.values() for item in bindings.values()]


def parse_hccn(text, device):
    """Read EID table and explicit port-status table; ignore IPoURMA IP rows."""
    eids, ports, name = {}, {}, None
    for line in text.splitlines():
        if not line.strip().startswith('|'):
            continue
        cells = [part.strip() for part in line.strip().strip('|').split('|')]
        if len(cells) == 6 and cells[0].isdigit() and cells[1].isdigit():
            key = (int(cells[0]), int(cells[1]))
            value = (cells[3], cells[4])
            if key in ports and ports[key] != value:
                raise ValueError(f'hccn device {device}: conflicting port status for {key}')
            ports[key] = value
        if len(cells) != 3:
            continue
        if cells[0].startswith('udmac'):
            name = cells[0]
        elif cells[0]:
            name = None
        if name is None or not cells[1].isdigit():
            continue
        eid = normalize_eid(cells[2])
        value = dict(name=name, index=int(cells[1]))
        if eid in eids and eids[eid] != value:
            raise ValueError(f'hccn device {device}: EID appears on multiple endpoints: {eid}')
        eids[eid] = value
    if not eids or not ports:
        raise ValueError(f'hccn device {device}: missing EID table or explicit port-status table')
    return eids, ports


def add_machine_arguments(parser):
    parser.add_argument('--root-info', '--rootinfo', type=Path,
                        help='current-machine version 2.0 rootinfo; auto-detect /etc/hccl_root[iI]nfo.json')
    parser.add_argument('--topology-json', type=Path,
                        help='current Atlas topology; auto-detect atlas_950.json or atlas_950_1.json')
    parser.add_argument('--hccn-tool', '--hccl-tool', default=HCCN_TOOL,
                        help='hccn_tool executable (read-only -g -dev_info -i PHYSICAL_ID)')
    parser.add_argument('--query-timeout', type=float, default=15.0)
    parser.add_argument('--net-layer', type=int, default=0)
    parser.add_argument('--net-instance-id', help='select one machine from a multi-instance rootinfo')
    parser.add_argument('--plane-id', help='rootinfo network plane label; NOT an IO die number')
    parser.add_argument('--diagnostics-json', type=Path, help='write automatically collected provenance')
    parser.add_argument('--topology', type=Path, help='removed TXT input; rejected with migration instructions')


def load_machine(args, devices):
    if args.topology is not None:
        raise ValueError('--topology TXT inventory is no longer supported; use --root-info, '
                         '--topology-json and live --hccn-tool')
    if not 0 <= args.net_layer <= 7 or not 0 < args.query_timeout <= 300:
        raise ValueError('net-layer must be 0..7; query-timeout must be in (0, 300] seconds')
    root = select_json_path(args.root_info, ROOTINFO_PATHS, '--root-info')
    atlas = select_json_path(args.topology_json, ATLAS_PATHS, '--topology-json')
    peers, edges = load_topology(atlas)
    devices = set(devices)
    if not devices <= peers:
        raise ValueError(f'Atlas topology missing physical IDs {sorted(devices - peers)}')
    entries = root_bindings(json.loads(root.read_text()), devices, args.net_layer,
                            args.net_instance_id, args.plane_id)
    inventory = {}
    queries = []
    for device in sorted(devices):
        cmd = [str(args.hccn_tool), '-g', '-dev_info', '-i', str(device)]
        try:
            result = subprocess.run(cmd, text=True, capture_output=True,
                                    timeout=args.query_timeout, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, 'stderr', '') or str(exc)
            raise ValueError(f'hccn query failed for physical device {device}: {detail}') from exc
        inventory[device] = parse_hccn(result.stdout, device)
        queries.append(dict(device=device, command=cmd,
                            stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest()))
    # Keep invalid/unavailable bindings for diagnostics, but never route through them.
    for item in entries:
        eids, ports = inventory[item['device']]
        live = eids.get(item['eid'])
        state = ports.get((item['die'], item['port']))
        item['verified'] = live is not None and state == ('UB', 'UP')
        item['name'] = live['name'] if live else '-'
        item['index'] = live['index'] if live else -1
        item['live_eid_present'] = live is not None
        item['port_state'] = state
    diagnostics = dict(version=1, source='rootinfo+atlas+live_hccn',
                       root_info=str(root.resolve()), topology_json=str(atlas.resolve()),
                       root_info_sha256=hashlib.sha256(root.read_bytes()).hexdigest(),
                       topology_json_sha256=hashlib.sha256(atlas.read_bytes()).hexdigest(),
                       queries=queries, bindings=entries)
    if args.diagnostics_json:
        args.diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_json.write_text(json.dumps(diagnostics, indent=2) + '\n')
    print(f'MACHINE_TOPOLOGY rootinfo={root} atlas={atlas} devices={sorted(devices)}', file=sys.stderr)
    return entries, edges, diagnostics


def endpoint_eids(entries, device, die, port, net_layer=0):
    return [item for item in entries if item['device'] == device and item['die'] == die
            and item['port'] == port and item['net_layer'] == net_layer and item['verified']]


def connected_pair(entries, a, a_port, b, b_port, net_layer):
    for left in endpoint_eids(entries, a, *a_port, net_layer):
        for right in endpoint_eids(entries, b, *b_port, net_layer):
            # The explicit Atlas edge connects the two local ports. Their
            # rootinfo plane labels need not match across devices/dies (e.g.
            # device 2 die0 plane_0 <-> relay 4 die1 plane_1).
            if left['net_instance_id'] == right['net_instance_id']:
                yield left, right


def relay_candidates(entries, edges, src, dst, relay, plane, net_layer):
    if len({src, dst, relay}) != 3:
        raise ValueError('source, destination and relay physical IDs must be distinct')
    rows = []
    for a in directed_links(edges, src, relay, net_layer):
        for b in directed_links(edges, dst, relay, net_layer):
            sd, sp = a['endpoint_port']; dd, dp = b['endpoint_port']
            rd, rp = a['relay_port']; rd2, rp2 = b['relay_port']
            if rd != rd2 or (plane != 'all' and rd != int(plane)):
                continue
            for se, ri in connected_pair(entries, src, a['endpoint_port'], relay, a['relay_port'], net_layer):
                for de, ro in connected_pair(entries, dst, b['endpoint_port'], relay, b['relay_port'], net_layer):
                    if ri['plane_id'] != ro['plane_id']:
                        continue
                    rows.append(dict(relay_phy=relay, plane=rd, src_die=sd, dst_die=dd,
                        relay_die=rd, src_port=sp, relay_port_from_src=rp,
                        relay_port_to_dst=rp2, dst_port=dp, src_udmac=se['name'],
                        dst_udmac=de['name'], src_eid_idx=se['index'], dst_eid_idx=de['index'],
                        src_eid=se['eid'], dst_eid=de['eid'], src_edge=a['edge'], dst_edge=b['edge'],
                        protocols=','.join(sorted(set(a['protocols']) & set(b['protocols'])))))
    # Duplicate topology rows must not silently select a different physical route.
    unique = {tuple(sorted(row.items())): row for row in rows}
    return list(unique.values())
