#!/usr/bin/env python3
"""Offline fixtures + fake read-only hccn process; never initialize an NPU."""
import argparse
import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from a5_ccu_machine_topology import (
    add_machine_arguments, load_machine, normalize_eid, parse_hccn, root_bindings,
    load_topology, select_json_path, relay_candidates,
)
from resolve_a5_explicit_multirelay_eids import resolve_relay
from prepare_a5_ccu_peer_plan import build_rows


def fixture():
    ranks, edges, live = [], [], {}
    for device in range(8):
        addresses, lines = [], []
        for die in (0, 1):
            for port in range(8):
                # Address bits intentionally DO NOT encode the port. UDMAC
                # instance/index intentionally DO NOT encode physical device.
                eid = f'0000:0000:0400:0300:0010:0000:df83:{1 + device * 32 + die * 8 + port:04x}'
                addresses.append(dict(addr_type='EID', addr=eid.replace(':', ''),
                                      ports=[f'{die}/{port}'], plane_id=f'fabric-{die + 20}'))
                lines += [f'| {die} | {port} | 400 | UB | UP | Electrical |',
                          f'| udmac0d{die}e3 | {port + 9} | {eid} |']
        ranks.append(dict(rank_id=7 - device, device_id=device, local_id=device + 20,
                          level_list=[dict(net_layer=0, net_instance_id='this-machine',
                                          net_type='TOPO_FILE_DESC', rank_addr_list=addresses)]))
        live[device] = '\n'.join(lines) + '\n'
    for a in range(8):
        for b in range(a + 1, 8):
            edges.append(dict(net_layer=0, link_type='PEER2PEER', protocols=['UB_CTP'],
                              local_a=a, local_b=b,
                              local_a_ports=[f'0/{b}', f'1/{b}'],
                              local_b_ports=[f'0/{a}', f'1/{a}']))
    return (dict(version='2.0', status='completed', rank_count=8, rank_list=ranks),
            dict(version='2.0', peer_count=8, peer_list=[dict(local_id=i) for i in range(8)],
                 edge_count=len(edges), edge_list=edges), live)


class MachineTopologyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.root, self.atlas, self.live = fixture()
        self.root_path = self.path / 'hccl_rootInfo.json'
        self.atlas_path = self.path / 'atlas_950.json'
        self.tool = self.path / 'hccn_tool'
        self.tool.write_text(
            '#!' + sys.executable + '\n'
            'import json, pathlib, sys\n'
            'p = pathlib.Path(__file__).parent\n'
            'assert sys.argv[1:4] == ["-g", "-dev_info", "-i"]\n'
            'assert len(sys.argv) == 5\n'
            'with (p / "queries").open("a") as f: f.write(sys.argv[4] + "\\n")\n'
            'print(json.loads((p / "live.json").read_text())[sys.argv[4]])\n')
        self.tool.chmod(0o755)
        self.save()
        parser = argparse.ArgumentParser()
        add_machine_arguments(parser)
        self.args = parser.parse_args(self.options())

    def save(self):
        self.root_path.write_text(json.dumps(self.root))
        self.atlas_path.write_text(json.dumps(self.atlas))
        (self.path / 'live.json').write_text(json.dumps(self.live))

    def options(self):
        return ['--root-info', str(self.root_path), '--topology-json', str(self.atlas_path),
                '--hccn-tool', str(self.tool)]

    def run_script(self, script, *args):
        return subprocess.run([sys.executable, str(ROOT / 'scripts' / script),
                               *self.options(), *args], capture_output=True, text=True, timeout=20)

    def loaded(self, devices=(2, 3, 4)):
        self.save()
        return load_machine(self.args, devices)

    def test_explicit_join_new_encoding_and_physical_id(self):
        entries, edges, provenance = self.loaded()
        row = resolve_relay(entries, edges, 2, 3, 4, '0', 0)
        self.assertEqual(row['src_port'], 4)
        self.assertEqual(row['relay_port_from_src'], 2)
        self.assertEqual(row['relay_port_to_dst'], 3)
        self.assertEqual(row['dst_port'], 4)
        self.assertEqual(row['src_eid'], normalize_eid(self.root['rank_list'][2]['level_list'][0]['rank_addr_list'][4]['addr']))
        self.assertEqual(row['src_udmac'], 'udmac0d0e3')
        self.assertEqual(row['src_eid_idx'], 13)  # index != port
        self.assertEqual(provenance['source'], 'rootinfo+atlas+live_hccn')
        self.assertEqual((self.path / 'queries').read_text().splitlines(), ['2', '3', '4'])
        reverse = resolve_relay(entries, edges, 3, 2, 4, '0', 0)
        self.assertEqual(reverse['src_eid'], row['dst_eid'])
        self.assertEqual(reverse['dst_eid'], row['src_eid'])

    def test_resource_rootinfo_without_rank_or_local_ids(self):
        expected, _, _ = self.loaded()
        for rank in self.root['rank_list']:
            rank.pop('rank_id')
            rank.pop('local_id')
        entries, edges, _ = self.loaded()
        before = resolve_relay(expected, edges, 2, 3, 4, '0', 0)
        after = resolve_relay(entries, edges, 2, 3, 4, '0', 0)
        self.assertEqual(before, after)
        # Exercise the actual CLI too, rather than only the parser function.
        result = self.run_script('resolve_a5_explicit_multirelay_eids.py',
                                 '--src-phy', '2', '--dst-phy', '3',
                                 '--relay-phys', '4', '--relay-planes', '0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(result.stdout), delimiter='\t'))), 1)

    def test_missing_required_fields_have_location_and_keys(self):
        original = copy.deepcopy(self.root)
        for field in ('device_id', 'level_list'):
            data = copy.deepcopy(original)
            del data['rank_list'][2][field]
            with self.assertRaisesRegex(ValueError, rf'rank_list\[2\].{field}; available keys:'):
                root_bindings(data, {2, 3, 4})
        data = copy.deepcopy(original)
        address = data['rank_list'][2]['level_list'][0]['rank_addr_list'][0]
        del address['ports']
        with self.assertRaisesRegex(ValueError, r'rank_addr_list\[0\].ports; available keys:'):
            root_bindings(data, {2, 3, 4})

    def test_eid_formats_and_invalid_values(self):
        eid = '0000:0000:0400:0300:0010:0000:df83:4152'
        for value in (eid, eid.upper(), eid.replace(':', ''), '::400:300:10:0:df83:4152'):
            self.assertEqual(normalize_eid(value), eid)
        for value in ('', '::', '0000', '1.2.3.4', 'fe80::1%eth0', None):
            with self.assertRaises(ValueError):
                normalize_eid(value)

    def test_new_machine_hccn_row_format(self):
        text = '''| 0 | 0 | 400 | UB | UP | Electrical |
| Name | EID Index | EID |
| udmac0d0e3 | 0 | 0000:0000:0400:0300:0010:0000:df83:4152 |
| | 1 | 0000:0000:043f:0300:0010:0000:df83:4c52 |
| ipourma1 | ::400:300:10:0:df83:4152 | ffff:ffff:ffff:ffff:: | NA | NA |
'''
        eids, ports = parse_hccn(text, 2)
        self.assertEqual(len(eids), 2)
        self.assertEqual(eids[normalize_eid('::43f:300:10:0:df83:4c52')]['index'], 1)
        self.assertEqual(ports[(0, 0)], ('UB', 'UP'))

    def test_multiport_address_not_address_bit_decoding(self):
        addresses = self.root['rank_list'][2]['level_list'][0]['rank_addr_list']
        addresses[3]['ports'].append('0/4')
        del addresses[4]
        entries, edges, _ = self.loaded()
        row = resolve_relay(entries, edges, 2, 3, 4, '0', 0)
        self.assertEqual(row['src_eid'], normalize_eid(addresses[3]['addr']))
        self.assertEqual(row['src_port'], 4)

    def test_missing_stale_eid_and_down_port_on_each_hop(self):
        original = copy.deepcopy(self.live)
        # Source, destination, relay ingress AND relay egress are all checked.
        for device, port in ((2, 4), (3, 4), (4, 2), (4, 3)):
            for failure in ('eid', 'port'):
                self.live = copy.deepcopy(original)
                if failure == 'eid':
                    eid = normalize_eid(self.root['rank_list'][device]['level_list'][0]['rank_addr_list'][port]['addr'])
                    self.live[device] = self.live[device].replace(eid, '0000:0000:9999:0000:0000:0000:0000:0001')
                else:
                    self.live[device] = self.live[device].replace(
                        f'| 0 | {port} | 400 | UB | UP |', f'| 0 | {port} | 400 | UB | DOWN |')
                entries, edges, _ = self.loaded()
                with self.subTest(device=device, port=port, failure=failure):
                    self.assertEqual(relay_candidates(entries, edges, 2, 3, 4, '0', 0), [])

    def test_atlas_cross_die_hop_with_different_endpoint_plane_labels(self):
        # Actual user topology: 2:0/6 -> 4:1/0 -> 3:0/3.
        for device, die, port, plane in (
            (2, 0, 6, 'plane_0'), (3, 0, 3, 'plane_0'),
            (4, 1, 0, 'plane_1'), (4, 1, 3, 'plane_1'),
        ):
            addresses = self.root['rank_list'][device]['level_list'][0]['rank_addr_list']
            addresses[die * 8 + port]['plane_id'] = plane
        self.atlas['edge_list'] = [
            dict(net_layer=0, link_type='PEER2PEER', protocols=['UB_CTP'],
                 local_a=2, local_b=4, local_a_ports=['0/6'], local_b_ports=['1/0']),
            dict(net_layer=0, link_type='PEER2PEER', protocols=['UB_CTP'],
                 local_a=3, local_b=4, local_a_ports=['0/3'], local_b_ports=['1/3']),
        ]
        self.atlas['edge_count'] = 2
        entries, edges, _ = self.loaded()
        row = resolve_relay(entries, edges, 2, 3, 4, 'all', 0)
        self.assertEqual((row['src_die'], row['relay_die'], row['dst_die']), (0, 1, 0))
        self.assertEqual((row['src_port'], row['relay_port_from_src'],
                          row['relay_port_to_dst'], row['dst_port']), (6, 0, 3, 3))
        self.assertEqual(relay_candidates(entries, edges, 2, 3, 4, '0', 0), [])
        # Removing plane-label equality MUST NOT bypass absent live EIDs.
        for device, die, port in ((2, 0, 6), (3, 0, 3), (4, 1, 0), (4, 1, 3)):
            altered = copy.deepcopy(entries)
            for entry in altered:
                if (entry['device'], entry['die'], entry['port']) == (device, die, port):
                    entry['verified'] = False
                    entry['live_eid_present'] = False
            self.assertEqual(relay_candidates(altered, edges, 2, 3, 4, 'all', 0), [])
        result = self.run_script('resolve_a5_explicit_multirelay_eids.py',
                                 '--src-phy', '2', '--dst-phy', '3',
                                 '--relay-phys', '4', '--relay-planes', '1')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plane_label_is_not_die_and_mismatch_is_rejected(self):
        self.args.plane_id = 'fabric-21'
        entries, edges, _ = self.loaded()
        self.assertEqual(resolve_relay(entries, edges, 2, 3, 4, 'all', 0)['relay_die'], 1)
        self.root['rank_list'][4]['level_list'][0]['rank_addr_list'][10]['plane_id'] = 'other'
        entries, edges, _ = self.loaded()
        self.assertEqual(relay_candidates(entries, edges, 2, 3, 4, 'all', 0), [])

    def test_ambiguous_path_requires_explicit_selection(self):
        entries, edges, _ = self.loaded()
        with self.assertRaisesRegex(SystemExit, '2 candidates'):
            resolve_relay(entries, edges, 2, 3, 4, 'all', 0)
        self.assertEqual(resolve_relay(entries, edges, 2, 3, 4, '1', 0)['relay_die'], 1)

    def test_cross_die_relay_or_wrong_protocol_rejected(self):
        entries, edges, _ = self.loaded()
        for edge in edges:
            if (edge['local_a'], edge['local_b']) == (2, 4):
                edge['local_a_ports'], edge['local_b_ports'] = ['0/4'], ['0/2']
            if (edge['local_a'], edge['local_b']) == (3, 4):
                edge['local_a_ports'], edge['local_b_ports'] = ['0/4'], ['1/3']
        self.assertEqual(relay_candidates(entries, edges, 2, 3, 4, 'all', 0), [])
        for edge in edges:
            edge['protocols'] = ['ROCE']
        self.assertEqual(relay_candidates(entries, edges, 2, 3, 4, 'all', 0), [])

    def test_bad_rootinfo_contracts(self):
        original = copy.deepcopy(self.root)
        mutations = [
            lambda r: r.update(version='1.0'),
            lambda r: r.update(status='initializing'),
            lambda r: r.update(rank_count=9),
            lambda r: r['rank_list'][2].update(device_id=99),
            lambda r: r['rank_list'][2]['level_list'][0].update(net_type='CLOS'),
            lambda r: r['rank_list'][2]['level_list'][0].update(net_instance_id='other-machine'),
            lambda r: r['rank_list'][2]['level_list'][0].update(net_layer=1),
        ]
        for mutate in mutations:
            data = copy.deepcopy(original); mutate(data)
            with self.assertRaises(ValueError):
                root_bindings(data, {2, 3, 4})
        addresses = self.root['rank_list'][2]['level_list'][0]['rank_addr_list']
        addresses.append(dict(addresses[0], addr='::1234'))
        with self.assertRaisesRegex(ValueError, 'ambiguous rootinfo binding'):
            root_bindings(self.root, {2, 3, 4})

    def test_duplicate_device_needs_instance_selector(self):
        remote = copy.deepcopy(self.root['rank_list'])
        for r in remote:
            r['rank_id'] += 8
            r['level_list'][0]['net_instance_id'] = 'other-machine'
        self.root['rank_list'] += remote
        self.root['rank_count'] = 16
        with self.assertRaisesRegex(ValueError, 'ambiguous device'):
            root_bindings(self.root, {2, 3, 4})
        self.assertEqual(len(root_bindings(self.root, {2, 3, 4}, net_instance_id='this-machine')), 48)

    def test_ipv6_host_addresses_are_not_reinterpreted(self):
        for item in self.root['rank_list'][2]['level_list'][0]['rank_addr_list']:
            item['addr_type'] = 'IPV6'
        with self.assertRaisesRegex(ValueError, 'no explicit EID'):
            root_bindings(self.root, {2, 3, 4})

    def test_topology_counts_ports_and_identity(self):
        original = copy.deepcopy(self.atlas)
        for mutate in (
            lambda t: t.update(edge_count=99),
            lambda t: t.update(peer_count=9),
            lambda t: t['edge_list'][0].update(local_b=99),
            lambda t: t['edge_list'][0].update(local_a_ports=['port0']),
            lambda t: t['edge_list'][0].update(local_a_ports=['0/1']),
        ):
            self.atlas = copy.deepcopy(original); mutate(self.atlas); self.save()
            with self.assertRaises(ValueError):
                load_topology(self.atlas_path)

    def test_automatic_path_detection_conflicts(self):
        other = self.path / 'second-rootinfo.json'  # distinct on case-insensitive macOS too
        other.write_text(self.root_path.read_text())
        self.assertEqual(select_json_path(None, [self.root_path, other], '--root-info'), self.root_path)
        other.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            select_json_path(None, [self.root_path, other], '--root-info')
        with self.assertRaisesRegex(ValueError, 'no current-machine'):
            select_json_path(None, [self.path / 'missing'], '--root-info')

    def test_live_query_failure_and_timeout_are_fatal(self):
        for failure in (FileNotFoundError('missing hccn'),
                        subprocess.TimeoutExpired('hccn', 1),
                        subprocess.CalledProcessError(2, 'hccn', stderr='driver query error')):
            with patch('a5_ccu_machine_topology.subprocess.run', side_effect=failure):
                with self.assertRaisesRegex(ValueError, 'hccn query failed'):
                    self.loaded()

    def test_peer_plan_two_and_four_rank_cli(self):
        result = self.run_script('prepare_a5_ccu_peer_plan.py', '--devices', '3,2',
                                 '--relay-map', str(self.write_policy({'2,3': [dict(physical_device=4, plane='0')]})),
                                 '--output-dir', str(self.path / 'two'))
        self.assertEqual(result.returncode, 0, result.stderr)
        two = json.loads((self.path / 'two/peer_plan.json').read_text())
        self.assertEqual(two['channels_per_rank'], [2, 2])
        self.assertEqual(two['rows'][1]['src_phy'], 3)
        policy = {'0,1': [4], '2,3': [4], '0,2': [5], '1,3': [5], '0,3': [6], '1,2': [6]}
        policy = {key: [dict(physical_device=values[0], plane='0')] for key, values in policy.items()}
        result = self.run_script('prepare_a5_ccu_peer_plan.py', '--devices', '0,1,2,3',
                                 '--relay-map', str(self.write_policy(policy)),
                                 '--output-dir', str(self.path / 'four'))
        self.assertEqual(result.returncode, 0, result.stderr)
        four = json.loads((self.path / 'four/peer_plan.json').read_text())
        self.assertEqual(four['channels_per_rank'], [6] * 4)
        self.assertEqual(len(four['rows']), 12)
        self.assertEqual(len(four['machine_topology']['queries']), 7)
        self.assertTrue(all(row['weight'] == (2 if row['kind'] == 'direct' else 1) for row in four['rows']))

    def write_policy(self, policy):
        p = self.path / 'policy.json'; p.write_text(json.dumps(policy)); return p

    def test_legacy_multirelay_tsv_columns_and_order(self):
        result = self.run_script('resolve_a5_explicit_multirelay_eids.py', '--src-phy', '2',
                                 '--dst-phy', '3', '--relay-phys', '7,6,5,4,1,0',
                                 '--relay-planes', '0,0,0,0,0,0', '--weights', '1,1,1,1,1,1')
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = list(csv.DictReader(io.StringIO(result.stdout), delimiter='\t'))
        self.assertEqual([int(r['relay_phy']) for r in rows], [7, 6, 5, 4, 1, 0])
        self.assertTrue(all(r['weight'] == '1' for r in rows))
        self.assertEqual(len((self.path / 'queries').read_text().splitlines()), 8)

    def test_synthetic_probe_cli_and_diagnostics(self):
        diagnostic = self.path / 'provenance.json'
        result = self.run_script('resolve_a5_synthetic_relay_eids.py', '--src-phy', '2',
                                 '--dst-phy', '3', '--relay-phy', '4', '--plane', '0',
                                 '--diagnostics-json', str(diagnostic))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(result.stdout), delimiter='\t'))), 1)
        self.assertEqual(len(json.loads(diagnostic.read_text())['queries']), 3)

    def test_direct_eight_rank_no_txt_or_hccn_requirement(self):
        result = self.run_script('prepare_a5_ccu_peer_plan.py', '--devices', '7,6,5,4,3,2,1,0',
                                 '--output-dir', str(self.path / 'eight'))
        self.assertEqual(result.returncode, 0, result.stderr)
        eight = json.loads((self.path / 'eight/peer_plan.json').read_text())
        self.assertEqual(eight['channels_per_rank'], [7] * 8)
        self.assertEqual(len(eight['rows']), 28)
        self.assertFalse((self.path / 'queries').exists())

    def test_old_txt_rejected_without_reading_or_querying(self):
        result = self.run_script('prepare_a5_ccu_peer_plan.py', '--devices', '2,3', '--relay-phys', '4',
                                 '--topology', '/must-not-read/old.txt', '--output-dir', str(self.path / 'bad'))
        self.assertEqual(result.returncode, 2)
        self.assertIn('TXT', result.stderr)
        self.assertFalse((self.path / 'bad').exists())
        self.assertFalse((self.path / 'queries').exists())

    def test_launchers_forward_machine_options_and_have_no_static_inventory(self):
        for name in ('peer_plan_alltoall', 'explicit_multirelay_alltoall',
                     'explicit_multipath_alltoall_perf', 'synthetic_relay_probe',
                     'synthetic_relay_causal_compare', 'hbm_command_puncture'):
            path = ROOT / f'scripts/run_a5_ccu_{name}.sh'
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('a5_hccn_device_topology_raw.txt', path.read_text())
            self.assertIn('"${machine_args[@]}"', path.read_text())


if __name__ == '__main__':
    unittest.main()
