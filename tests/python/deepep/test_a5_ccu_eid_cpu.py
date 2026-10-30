"""EP8 participant-EID catalog, production CCU graph and C++ host checks."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_a5_ccu_ep8_topology import build_eid_topology

class EidTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'), 'g++ required')
    def test_production_ccu_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = str(Path(tmp) / 'graph')
            subprocess.run(['g++', '-std=c++14', '-Wall', '-Wextra', '-Werror',
                '-I' + str(ROOT / 'tests/cpp/fake_ccu'),
                str(ROOT / 'tests/cpp/test_a5_ccu_eid.cpp'),
                str(ROOT / 'examples/a5_ccu_urma_route_probe/op_kernel_ccu/eid_alltoallv_kernel.cc'),
                str(ROOT / 'tests/cpp/group_copy_model_stub.cc'), '-o', binary], check=True, timeout=90)
            result = subprocess.run([binary], capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('EP8_EID_GRAPH_PASS epochs=100 launches=800', result.stdout)
            print(result.stdout.strip())

    @staticmethod
    def machine():
        devices = [7, 2, 5, 0, 4, 6, 1, 3]
        entries, edges = [], []
        for s in range(8):
            for d in range(s + 1, 8):
                edges.append(dict(net_layer=0, link_type='PEER2PEER', protocols=['UB_CTP'],
                                  local_a=devices[s], local_b=devices[d],
                                  local_a_ports=[f'0/{d}'], local_b_ports=[f'0/{s}']))
                for a,b in ((s,d),(d,s)):
                    entries.append(dict(device=devices[a], die=0, port=b, net_layer=0,
                                        net_instance_id='test', plane_id='plane_0', name='test', index=b,
                                        eid=f'{a}-{b}', verified=True))
        return devices, entries, edges

    def test_live_catalog_group_relays_and_physical_rank_mapping(self):
        devices, entries, edges = self.machine()
        rows,t = build_eid_topology(devices, entries, edges)
        self.assertEqual(t['executor'], 'participant_eid_ccu_urma')
        self.assertEqual(len(rows),196)
        self.assertEqual(t['channels_per_rank'],[49]*8)
        for row in rows:
            if row['kind'] != 'relay': continue
            s,d = row['src_rank'],row['dst_rank']; k=devices.index(row['relay_phy'])
            self.assertNotIn(k,(s,d)); self.assertIn(k,t['allowed_relays'][s*8+d])
            self.assertEqual(row['src_eid'],f'{s}-{k}')
            self.assertEqual(row['dst_eid'],f'{d}-{k}')
        limited,t = build_eid_topology(devices, entries, edges,max_relays=1)
        self.assertEqual(len(limited),56);self.assertEqual(t['channels_per_rank'],[14]*8)
        entries[0]['verified']=False
        with self.assertRaises(ValueError): build_eid_topology(devices,entries,edges)

    def test_same_relay_plane_required_no_guessed_candidates(self):
        devices,entries,edges=self.machine()
        # Keep both direct links valid, but split the two relay-side planes.
        for e in entries:
            if e['device']==devices[2] and e['port']==1: e['plane_id']='other'
        _,t=build_eid_topology(devices,entries,edges)
        self.assertNotIn(2,t['allowed_relays'][1])
        self.assertNotIn(2,t['allowed_relays'][8])
        self.assertTrue(t['excluded_relays'])

    @unittest.skipUnless(importlib.util.find_spec('torch') and shutil.which('g++'), 'Torch and g++ required')
    def test_production_host_dual_dtype_single_submission(self):
        import torch
        tr=Path(torch.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            binary=str(Path(tmp)/'host')
            subprocess.run(['g++','-std=c++17','-O1',
                '-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch.compiled_with_cxx11_abi())),
                '-I'+str(tr/'include'),'-I'+str(tr/'include/torch/csrc/api/include'),
                str(ROOT/'tests/cpp/test_a5_ccu_eid_host.cpp'),
                '-L'+str(tr/'lib'),'-Wl,-rpath,'+str(tr/'lib'),'-ltorch_cpu','-lc10','-o',binary],check=True,timeout=180)
            result=subprocess.run([binary],capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('EP8_EID_HOST_PASS',result.stdout)
            print(result.stdout.strip())

if __name__=='__main__': unittest.main()
