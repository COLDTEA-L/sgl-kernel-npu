"""Machine-bound L1 policy, actual planner bindings and shared port budgets."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from a5_ccu_l1_policy import single_port, sha, apply_l1, validate_config


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.dir=Path(self.tmp.name)
        self.root=self.dir/'root.json';self.inv=self.dir/'inventory.json';self.config=self.dir/'policy.json'
        self.eids=['00000000040102000010000000000001','00000000040102000010000000000002']
        self.root.write_text(json.dumps(dict(version='2.0',rank_list=[dict(device_id=d,level_list=[dict(net_layer=1,net_type='CLOS',net_instance_id='sp',rank_addr_list=[dict(addr_type='EID',addr=self.eids[0].replace('0401','043f'),plane_id='pg',ports=['0/1','0/2'])])]) for d in (2,3)])))
        self.inv.write_text(json.dumps(dict(schema='a5.endpoint-inventory.v1',rootinfo_sha256=sha(self.root),endpoints=[dict(device_id=d,protocol=4,addr_type=1,eid=e) for d,e in zip((2,3),self.eids)])))
        self.c=dict(schema='a5.single-port-l1.v1',generation='controller-1',devices=[2,3],rootinfo=str(self.root),rootinfo_sha256=sha(self.root),inventory='inventory.json',inventory_sha256=sha(self.inv),paths=[dict(id='pair',src_phy=2,dst_phy=3,src_eid=self.eids[0],dst_eid=self.eids[1],src_die=0,dst_die=0,src_port=1,dst_port=1,plane_id='pg',net_instance_id='sp',l0_weight=1,l1_weight=1)])
        self.save()
    def save(self): self.config.write_text(json.dumps(self.c))
    def test_decode_is_not_fixed_group_width(self):
        self.assertEqual(single_port(self.eids[0]),(0,1))
        self.assertEqual(single_port(self.eids[0].replace('0401','0441')),(1,1))
        for prefix in ('043f','047f'):
            with self.assertRaises(ValueError):single_port(self.eids[0].replace('0401',prefix))
    def test_apply_and_controller_generation(self):
        validate_config(self.config,[2,3])
        direct=dict(src_rank=0,dst_rank=1,src_phy=2,dst_phy=3,kind='direct',relay_phy='-',direct_route=0,weight=2,src_die=0,dst_die=0,src_eid=self.eids[0].replace('0401','0407'),dst_eid=self.eids[1].replace('0401','0400'))
        rows,p=apply_l1([direct],[2,3],self.config)
        self.assertEqual([r['kind'] for r in rows],['direct','l1_single'])
        self.assertEqual(p['l1_resources'],[[],[1,19],[19,1],[]])
        self.assertEqual(p['generation'],'controller-1')
        self.assertFalse(p['physical_single_port_verified'])
        self.assertEqual(direct['weight'],2)
    def test_reject_unobserved_wrong_membership_mapping_and_stale(self):
        original=copy.deepcopy(self.c)
        for k,v in [('src_eid',self.eids[0][:-1]+'f'),('src_port',2),('src_die',1),('plane_id','wrong'),('l1_weight',0)]:
            self.c=copy.deepcopy(original);self.c['paths'][0][k]=v;self.save()
            with self.assertRaises(ValueError):validate_config(self.config,[2,3])
        self.c=original;self.save()
        with self.assertRaises(ValueError):validate_config(self.config,[3,2])
        self.root.write_text(self.root.read_text()+' ')
        with self.assertRaises(ValueError):validate_config(self.config,[2,3])
    def test_resource_conservation_cpp(self):
        binary=self.dir/'planner'
        subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-Werror',str(ROOT/'tests/cpp/test_a5_ccu_l1_planner.cpp'),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True)


class BindingTests(unittest.TestCase):
    def test_actual_native_l1_binding_and_report(self):
        # Native pybind classes are global; isolate from other planner test modules.
        if not __import__('os').environ.get('A5_L1_BINDING_CHILD'):
            env=dict(__import__('os').environ,A5_L1_BINDING_CHILD='1')
            subprocess.run([sys.executable,__file__,'BindingTests.test_actual_native_l1_binding_and_report'],env=env,check=True)
            return
        from test_a5_ccu_native_planner_cpu import load_native_planner
        from heuristic_alltoallv import RelayPlanner
        with tempfile.TemporaryDirectory() as tmp:
            native=load_native_planner(tmp)
            ports=[[] for _ in range(64)];ports[1]=[1,19];ports[8]=[19,1]
            planner=RelayPlanner(5120,[[] for _ in range(64)],strategy='l1_bottleneck',l1_resources=ports,native_module=native)
            counts=[0]*64;counts[1]=1000
            plan=planner.plan(counts);report=plan.report()
            self.assertEqual(plan.relay_rank[1],8);self.assertEqual(plan.relay_rows[1],500)
            self.assertEqual(report['l1_rows'],500)
            self.assertEqual(report['transmitted_network_bytes'],1000*5120)
            self.assertEqual(report['planned_link_peak_bytes'],500*5120)
            with self.assertRaises(ValueError):RelayPlanner(5120,[[] for _ in range(64)],l1_resources=ports)


if __name__=='__main__':unittest.main()
