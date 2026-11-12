"""Graph-policy boundary: declared endpoints, sharing, stale files and mode gates."""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'python/deep_ep/deep_ep'))
from a5_ccu_graph_l1_policy import generate, validate, declared_endpoints
from a5_ccu_l1_policy import apply_l1, sha
from ccu_l1_config import select_l1_config, validate_l1_schema


class GraphL1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.root, self.base, self.policy = (self.dir/x for x in ('root.json','base.json','policy.json'))
        self.devices = list(range(8))
        self.dies = [0]*4 + [1]*4
        ranks = []
        self.rows = []
        for rank in range(8):
            # EID prefix does not encode the number of physical members.
            eid = f'00000000043f020000100000000000{rank:02x}'
            die = self.dies[rank]
            ranks.append(dict(device_id=rank,level_list=[dict(net_layer=1,net_type='CLOS',net_instance_id='sp',
                rank_addr_list=[dict(addr_type='EID',addr=eid,ports=[f'{die}/1',f'{die}/2'],plane_id='small')])]))
            for d in range(rank+1,8):
                self.rows.append(dict(src_rank=rank,dst_rank=d,src_phy=rank,dst_phy=d,kind='direct',
                    relay_phy='-',direct_route=0,weight=2,src_die=die,dst_die=self.dies[d],
                    src_eid=f'0000000004{die*64+7:02x}000000100000000000{rank:02x}',
                    dst_eid=f'0000000004{self.dies[d]*64+7:02x}000000100000000000{d:02x}'))
        self.data = dict(version='2.0',rank_list=ranks)
        self.save_root()
        with redirect_stdout(io.StringIO()):
            self.c = generate(self.root,self.base,self.policy)

    def save_root(self):
        self.root.write_text(json.dumps(self.data))
        self.base.write_text(json.dumps(dict(devices=self.devices,rank_dies=self.dies,
                                            provenance=dict(root_info_sha256=sha(self.root)))))

    def test_apply_shared_group_not_independent_peer_capacity(self):
        rows, meta = apply_l1(self.rows,self.devices,self.policy)
        self.assertEqual(len(rows),56)
        self.assertEqual(sum(r['kind']=='l1_graph' for r in rows),28)
        self.assertEqual(self.rows[0]['weight'],2)
        self.assertTrue(meta['runtime_graph_match_required'])
        for s in range(8):
            ids = {meta['l1_resources'][s*8+d][0] for d in range(8) if s!=d}
            self.assertEqual(ids,{s*18+self.dies[s]*9+1})
            for d in range(8):
                if s!=d:
                    self.assertEqual(meta['l1_resources'][s*8+d],meta['l1_resources'][d*8+s][::-1])
        self.assertEqual(set(meta['l1_budget']['tx_capacity_permille']),{1000})

    def test_reject_synthetic_eid_port_plane_and_mapping(self):
        for key,value in [('src_eid','00000000040102000010000000000000'),('src_ports',[1]),
                          ('src_die',1),('plane_id','fake')]:
            bad=copy.deepcopy(self.c);bad['paths'][0][key]=value
            with self.assertRaises(ValueError):validate(bad,self.devices,self.root)
        with self.assertRaises(ValueError):validate(self.c,self.devices[::-1],self.root)
        self.root.write_text(self.root.read_text()+' ')
        with self.assertRaises(ValueError):validate(self.c,self.devices,self.root)

    def test_no_overlap_and_no_cross_plane_fallback(self):
        bad=copy.deepcopy(self.data)
        a=bad['rank_list'][0]['level_list'][0]['rank_addr_list']
        a.append(dict(a[0],addr='00000000040102000010000000000000',ports=['0/1']))
        with self.assertRaises(ValueError):declared_endpoints(bad,self.devices)
        self.data['rank_list'][0]['level_list'][0]['rank_addr_list'][0]['plane_id']='other'
        self.save_root()
        with self.assertRaises(ValueError):generate(self.root,self.base,self.policy)

    def test_single_group_per_rank_and_duplicate_pair_rejected(self):
        bad=copy.deepcopy(self.c);bad['paths'].append(bad['paths'][0])
        with self.assertRaises(ValueError):validate(bad,self.devices,self.root)
        bad=copy.deepcopy(self.c);bad['resource_budget']={}
        with self.assertRaises(ValueError):validate(bad,self.devices,self.root)

    def test_overlap_l0_and_wrong_die_rejected(self):
        bad=copy.deepcopy(self.rows);bad[0]['src_eid']=bad[0]['src_eid'].replace('0407','0401')
        with self.assertRaises(ValueError):apply_l1(bad,self.devices,self.policy)
        bad=copy.deepcopy(self.rows);bad[0]['src_die']=1
        with self.assertRaises(ValueError):apply_l1(bad,self.devices,self.policy)

    def test_graph_mode_default_and_explicit_opt_in(self):
        with patch.dict(os.environ,{},clear=True), redirect_stdout(io.StringIO()):
            self.assertEqual(select_l1_config(self.policy),self.policy)
            with self.assertRaises(ValueError):validate_l1_schema('a5.single-port-l1.v1')
        with patch.dict(os.environ,{'A5_L1_ENDPOINT_MODE':'explicit_single'},clear=True):
            self.assertEqual(validate_l1_schema('a5.single-port-l1.v1'),'explicit_single')
            with self.assertRaises(ValueError):validate_l1_schema('a5.rank-graph-l1.v1')
        with patch.dict(os.environ,{'A5_ENABLE_L1':'0'},clear=True), redirect_stdout(io.StringIO()):
            self.assertIsNone(select_l1_config('/missing'))

    def test_prefer_small_common_group_and_honor_plane(self):
        for rank in self.data['rank_list']:
            a=rank['level_list'][0]['rank_addr_list']
            die=self.dies[rank['device_id']]
            a.append(dict(a[0],addr=a[0]['addr'].replace('043f','047f'),plane_id='large',
                          ports=[f'{die}/{p}' for p in (0,3,4,5,6)]))
        self.save_root()
        with redirect_stdout(io.StringIO()):
            c=generate(self.root,self.base,self.policy)
            self.assertEqual(c['paths'][0]['plane_id'],'small')
            c=generate(self.root,self.base,self.policy,plane='large')
            self.assertEqual(c['paths'][0]['plane_id'],'large')


if __name__=='__main__':unittest.main()
