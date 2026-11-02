"""Production C++ target planner: known solution, invariants and payload replay."""
import random
import tempfile
import unittest
from test_a5_ccu_native_planner_cpu import load_native_planner
from test_a5_ccu_pipeline_cpu import ALL, simulate
from heuristic_alltoallv import RelayPlanner, MAX_I64

REAL = (
42,539,744,205,1371,715,65,357,
57,529,727,293,1331,714,66,321,
31,411,906,311,1127,873,63,316,
24,548,869,125,1089,931,69,383,
29,495,775,167,1479,617,132,344,
424,347,755,117,1327,609,47,412,
35,559,779,200,1280,648,73,464,
72,569,728,165,1336,847,99,222,
)


class BottleneckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='ep8-bottleneck-')
        cls.native = load_native_planner(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def planner(self, rb=5120, allowed=ALL, **kwargs):
        return RelayPlanner(rb, allowed, native_module=self.native, strategy='bottleneck', **kwargs)

    def verify(self, p, allowed=ALL, caps=None, min_chunk=1, min_gain=0, limit=56):
        original = [n if f//8 != f%8 else 0 for f,n in enumerate(p.counts)]
        rebuilt = original[:]
        used = [0]*8
        self.assertLessEqual(len(p.actions), min(limit,8))
        self.assertLessEqual(p.candidate_evaluations, 4*56*6)
        for f,x in enumerate(p.relay_rows):
            k = p.relay_rank[f]
            if not x:
                self.assertEqual(k,-1)
                continue
            s,d = divmod(f,8)
            self.assertIn(k,allowed[f])
            self.assertLessEqual(x,p.counts[f])
            self.assertGreaterEqual(x*p.row_bytes,min_chunk)
            rebuilt[f]-=x;rebuilt[s*8+k]+=x;rebuilt[k*8+d]+=x
            used[k]+=x*p.row_bytes
        self.assertEqual(tuple(rebuilt),p.link_rows)
        self.assertEqual(sum(rebuilt),sum(original)+sum(p.relay_rows))
        self.assertEqual(len(p.actions),sum(x>0 for x in p.relay_rows))
        if caps: self.assertTrue(all(a<=b for a,b in zip(used,caps)))
        if p.uses_relay:
            self.assertLess(max(rebuilt),max(original))
            self.assertGreaterEqual((max(original)-max(rebuilt))*p.row_bytes,min_gain)
        else:
            self.assertEqual(rebuilt,original)

    def test_real_matrix_reduces_unnecessary_forwarding(self):
        p = self.planner().plan(REAL)
        self.verify(p)
        self.assertEqual(max(p.link_rows),1281)
        self.assertEqual(sum(p.relay_rows),241)
        self.assertEqual(len(p.actions),4)
        self.assertTrue(all(a[1]==4 for a in p.actions))
        old = RelayPlanner(5120,ALL,native_module=self.native).plan(REAL)
        self.assertLess(sum(p.relay_rows),sum(old.relay_rows))
        self.assertLessEqual(max(p.link_rows),max(old.link_rows))
        self.assertEqual(self.planner(10240).plan(REAL).relay_rows,p.relay_rows)
        simulate(p,5120*128)
        reverse = tuple(REAL[d*8+s] for s in range(8) for d in range(8))
        self.verify(self.planner(10240).plan(reverse))

    def test_random_topology_capacity_and_changed_counts(self):
        rng=random.Random(93217)
        for trial in range(350):
            c=[rng.randrange(80) for _ in range(64)]
            if trial%2: c[rng.randrange(64)]+=300
            allowed=[sorted(rng.sample(list(row),rng.randrange(len(row)+1))) for row in ALL]
            rb=rng.choice([1,8,5120,10240]);caps=[rng.randrange(100)*rb for _ in range(8)]
            opts=dict(min_chunk_bytes=rng.choice([1,rb,3*rb]),min_gain_bytes=rng.randrange(4)*rb,
                      max_actions=rng.choice([0,1,4,8,56]),relay_capacity_bytes=caps)
            planner=self.planner(rb,allowed,**opts)
            p=planner.plan(c)
            self.verify(p,allowed,caps,opts['min_chunk_bytes'],opts['min_gain_bytes'],opts['max_actions'])
            self.assertEqual(planner.plan(c),p)
            changed=list(c);changed[1]+=31
            self.verify(planner.plan(changed),allowed,caps,opts['min_chunk_bytes'],opts['min_gain_bytes'],opts['max_actions'])
            if trial<25: simulate(p,rb*32)

    def test_cut_uniform_zero_self_and_partial_trial_rollback(self):
        cut=[20 if s<4 and d>=4 else 0 for s in range(8) for d in range(8)]
        for c in ([0]*64,[10]*64,cut,[7 if s==d else 0 for s in range(8) for d in range(8)]):
            p=self.planner().plan(c);self.verify(p)
            self.assertFalse(p.uses_relay)
            self.assertEqual(p.candidate_evaluations,0)
        allowed=[list(row) for row in ALL];allowed[7*8+4]=[]
        p=self.planner(allowed=allowed).plan(REAL)
        self.verify(p,allowed)
        # Every trial still needs to lower 7->4, so no partial 0->4 move escapes.
        self.assertFalse(p.uses_relay)
        self.assertEqual(sum(p.relay_rows),0)
        self.assertFalse(self.planner(max_actions=1).plan(REAL).uses_relay)

    def test_single_hot_integer_limits_and_capabilities(self):
        for rows in (1,2,3,4096,MAX_I64):
            c=[0]*64;c[1]=rows
            p=self.planner(1).plan(c);self.verify(p)
            if rows>=2:
                self.assertEqual(max(p.link_rows),(rows+1)//2)
                self.assertEqual(sum(p.relay_rows),rows//2)
        c=[0]*64;c[1]=90
        self.assertFalse(self.planner(1,relay_capacity_bytes=[0]*8).plan(c).uses_relay)
        self.assertFalse(self.planner(1,min_gain_bytes=91).plan(c).uses_relay)
        with self.assertRaises(RuntimeError): RelayPlanner(1,ALL,strategy='bottleneck')
        with self.assertRaises(ValueError): RelayPlanner(1,ALL,strategy='invalid')
        c[0]=MAX_I64
        with self.assertRaises(ValueError): self.planner(1).plan(c)

if __name__=='__main__': unittest.main()
