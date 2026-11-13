"""Validate explicit relay hops against Atlas and live rootinfo/EID provenance."""
import hashlib
import json
from pathlib import Path

from a5_ccu_machine_topology import directed_links, load_topology


def link_key(a, ad, ap, b, bd, bp):
    left, right = sorted(((int(a), int(ad), int(ap)), (int(b), int(bd), int(bp))))
    return 'link:' + '<->'.join(f'{d}/{die}/{port}' for d, die, port in (left, right))


def row_links(row):
    s, d, r = (int(row[k]) for k in ('src_phy', 'dst_phy', 'relay_phy'))
    sd, dd, rd = (int(row[k]) for k in ('src_die', 'dst_die', 'relay_die'))
    return [link_key(s, sd, row['src_port'], r, rd, row['relay_port_from_src']),
            link_key(d, dd, row['dst_port'], r, rd, row['relay_port_to_dst'])]


class PhysicalInventory:
    def __init__(self, topology_path, provenance_path):
        self.peers, self.edges = load_topology(topology_path)
        self.provenance = json.loads(Path(provenance_path).read_text())
        p = self.provenance
        if (p.get('source') != 'rootinfo+atlas+live_hccn' or
                p.get('topology_json_sha256') != hashlib.sha256(Path(topology_path).read_bytes()).hexdigest()):
            raise ValueError('physical inventory must match live resolver topology provenance')
        self.layer = None

    def validate(self, row):
        p = self.provenance
        s, d, r = (int(row[k]) for k in ('src_phy', 'dst_phy', 'relay_phy'))
        if len({s, d, r}) != 3:
            raise ValueError('relay route has repeated physical card')
        # Endpoint EID selects an explicitly bound port. Reject an EID bound to
        # multiple ports: a logical EID alone would not prove this physical hop.
        from a5_ccu_machine_topology import normalize_eid
        layers, instances = set(), set()
        for side, device in [('src', s), ('dst', d)]:
            eid = normalize_eid(row[side + '_eid'])
            bindings = [b for b in p['bindings'] if b['device'] == device and b['eid'] == eid]
            ports = {(b['die'], b['port']) for b in bindings}
            expected = (int(row[side + '_die']), int(row[side + '_port']))
            if ports != {expected} or not all(b['verified'] for b in bindings):
                raise ValueError('EID is unavailable or does not uniquely select the manifest physical port')
            layers.update(b['net_layer'] for b in bindings)
            instances.update(b['net_instance_id'] for b in bindings)
        if len(layers) != 1 or len(instances) != 1:
            raise ValueError('route EIDs span layers/instances')
        layer = next(iter(layers))
        if self.layer is not None and self.layer != (layer, next(iter(instances))):
            raise ValueError('jobs span network layers/instances')
        self.layer = layer, next(iter(instances))
        rd = int(row['relay_die'])
        for side, device, relay_port in [('src', s, 'relay_port_from_src'), ('dst', d, 'relay_port_to_dst')]:
            expected = ((int(row[side + '_die']), int(row[side + '_port'])), (rd, int(row[relay_port])))
            matches = [link for link in directed_links(self.edges, device, r, layer)
                       if (link['endpoint_port'], link['relay_port']) == expected]
            if not matches:
                raise ValueError('manifest physical hop does not exist in Atlas UB_CTP topology')
            live = [b for b in p['bindings'] if b['device'] == r and
                    (b['die'], b['port']) == expected[1] and b['verified'] and b['net_layer'] == layer and
                    b['net_instance_id'] in instances]
            if not live:
                raise ValueError('relay physical port is not live/verified')
        return row_links(row)


def conflicts(jobs):
    result = []
    for i, a in enumerate(jobs):
        for b in jobs[i + 1:]:
            shared = sorted(set(a['payload_resources']) & set(b['payload_resources']))
            if shared:
                result.append(dict(jobs=[a['name'], b['name']], shared_payload_links=shared))
    return result
