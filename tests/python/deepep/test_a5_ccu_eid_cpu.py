"""EP8 participant-EID catalog, production CCU graph and C++ host checks."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from a5_ccu_startup_probe import traced_all_gather
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_a5_ccu_ep8_topology import build_eid_topology

class EidTests(unittest.TestCase):
    def test_create_buffer_default_and_explicit_override(self):
        clean = {k: v for k, v in os.environ.items() if k != 'HCCL_BUFFSIZE'}
        script = 'source "$1"; shift; a5_ccu_prepare_eid_buffer_env "$@"; printf "%s" "${HCCL_BUFFSIZE:-unset}"'
        for args, override, expected in (
            (['--suite', 'create'], None, '64'),
            (['--suite=create', '--minimal'], None, '64'),
            (['--suite', 'create'], '2300', '2300'),
            (['--suite', 'benchmark'], None, 'unset'),
            (['--suite', 'correctness'], None, 'unset'),
            ([], None, 'unset'),
        ):
            env = dict(clean)
            if override is not None: env['HCCL_BUFFSIZE'] = override
            result = subprocess.run(['bash', '-c', script, 'buffer-test',
                str(ROOT/'scripts/a5_ccu_test_env.sh')] + args, env=env,
                capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout, expected)

    def run_collective_probe(self, failure=None):
        events = []
        tensor = SimpleNamespace(dtype='torch.int32', shape=(32,), numel=lambda:32, element_size=lambda:4)
        wait = Mock(return_value=True)
        sync = Mock()
        if failure == 'wait': wait.side_effect = RuntimeError('wait rejected')
        if failure == 'device_sync': sync.side_effect = RuntimeError('stream failed')
        if failure == 'false_wait': wait.return_value = False
        npu = SimpleNamespace(current_stream=lambda:SimpleNamespace(npu_stream=123), synchronize=sync)
        dist = SimpleNamespace(all_gather=Mock(return_value=SimpleNamespace(wait=wait)))
        phase = lambda name, **fields: events.append((name, fields))
        def call(): traced_all_gather(SimpleNamespace(npu=npu), dist, [tensor]*8, tensor, phase)
        if failure:
            with self.assertRaises(RuntimeError): call()
        else: call()
        self.assertEqual(dist.all_gather.call_count, 1)
        self.assertEqual(dist.all_gather.call_args.kwargs, {'async_op': True})
        self.assertEqual(wait.call_count, 1)
        return events, sync

    def test_probe_distinguishes_wait_and_device_completion(self):
        events, sync = self.run_collective_probe()
        names = [name for name, fields in events]
        self.assertLess(names.index('config_allgather_wait_complete'), names.index('config_allgather_sync_begin'))
        self.assertEqual(names[-1], 'config_allgather_complete')
        self.assertEqual(sync.call_count, 1)
        self.assertEqual(events[0][1]['input_bytes'], 128)
        self.assertEqual(events[0][1]['output_bytes'], 1024)
        self.assertEqual(events[0][1]['caller_stream'], '123')

    def test_probe_failure_is_not_retried_or_reported_complete(self):
        for failure, stage in [('wait', 'wait'), ('false_wait', 'wait'), ('device_sync', 'device_sync')]:
            events, sync = self.run_collective_probe(failure)
            self.assertEqual(events[-1][0], 'config_allgather_failed')
            self.assertEqual(events[-1][1]['stage'], stage)
            self.assertNotIn('config_allgather_complete', [name for name, fields in events])
            self.assertEqual(sync.call_count, int(stage == 'device_sync'))

    def test_minimal_pass_is_not_registration_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'results').mkdir(); (root/'plans').mkdir()
            (root/'exit.status').write_text('0')
            (root/'plans/topology.json').write_text('{}')
            (root/'run.log').write_text(''.join(f'EP8_EID_MINIMAL_RANK_PASS rank={r}\n' for r in range(8)))
            for r in range(8):
                (root/'results'/f'rank{r}.json').write_text(json.dumps(dict(rank=r,
                    backend='participant_eid_ccu_urma', options=dict(suite='create', minimal=True), reports=[])))
            cmd = [sys.executable, str(ROOT/'scripts/summarize_a5_ccu_eid.py'), str(root)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertIn('EP8_EID_MINIMAL_ALL_RANKS_AND_EXIT_PASS', result.stdout)
            self.assertNotIn('EP8_EID_CREATE_ALL_RANKS_AND_EXIT_PASS', result.stdout)
            # Historical worker print calls can concatenate PASS records, or
            # prepend another worker's SDK output on the same line.
            (root/'run.log').write_text('[SDK] ' + ''.join(f'EP8_EID_MINIMAL_RANK_PASS rank={r}' for r in range(8)))
            merged = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertIn('EP8_EID_MINIMAL_ALL_RANKS_AND_EXIT_PASS', merged.stdout)

            # Exact field report: rank 7 is immediately followed by SDK INFO.
            clean = ''.join(f'EP8_EID_MINIMAL_RANK_PASS rank={r}\n' for r in range(7))
            for suffix in ('[INFO] ASCENDCL(27712,python3): device cleanup\n',
                           '[WARNING] SDK cleanup\n'):
                (root/'run.log').write_text(clean + 'EP8_EID_MINIMAL_RANK_PASS rank=7' + suffix)
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.assertIn('EP8_EID_MINIMAL_ALL_RANKS_AND_EXIT_PASS', result.stdout)
            # A missing rank or a two-digit rank must not be accepted as rank 7.
            for last in ('', 'EP8_EID_MINIMAL_RANK_PASS rank=70[INFO] SDK\n'):
                (root/'run.log').write_text(clean + last)
                rejected = subprocess.run(cmd, capture_output=True, text=True)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn('missing clean rank completion', rejected.stderr)
            (root/'run.log').write_text(clean + 'EP8_EID_MINIMAL_RANK_PASS rank=7[INFO] SDK\n')
            self.assertNotEqual(subprocess.run(cmd+['--compare',str(root)], capture_output=True).returncode,0)
            p = root/'results/rank3.json'
            record = json.loads(p.read_text()); record['options']['minimal'] = False
            p.write_text(json.dumps(record))
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode,0)

    def test_registration_probe_is_not_full_create_or_payload_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'results').mkdir(); (root/'plans').mkdir()
            (root/'exit.status').write_text('0'); (root/'plans/topology.json').write_text('{}')
            (root/'run.log').write_text(''.join(f'EP8_EID_REGISTER_PROBE_RANK_PASS rank={r}\n' for r in range(8)))
            for r in range(8):
                (root/'results'/f'rank{r}.json').write_text(json.dumps(dict(rank=r,
                    backend='participant_eid_ccu_urma', options=dict(suite='create',register_stage='args'), reports=[])))
            cmd=[sys.executable,str(ROOT/'scripts/summarize_a5_ccu_eid.py'),str(root)]
            result=subprocess.run(cmd,capture_output=True,text=True,check=True)
            self.assertIn('REGISTER_PROBE_ALL_RANKS_AND_EXIT_PASS stage=args',result.stdout)
            self.assertNotIn('EP8_EID_CREATE_ALL_RANKS_AND_EXIT_PASS',result.stdout)
            self.assertNotEqual(subprocess.run(cmd+['--compare',str(root)],capture_output=True).returncode,0)
            data=json.loads((root/'results/rank3.json').read_text());data['options']['register_stage']='mask'
            (root/'results/rank3.json').write_text(json.dumps(data))
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)

    def test_launcher_honors_sdk_logging_and_explicit_override(self):
        keys = ('A5_EID_LOG_LEVEL', 'A5_EID_LOG_STDOUT', 'ASCEND_GLOBAL_LOG_LEVEL',
                'ASCEND_MODULE_LOG_LEVEL', 'ASCEND_SLOG_PRINT_TO_STDOUT')
        clean = {k: v for k, v in os.environ.items() if k not in keys}
        script = 'source "$1"; a5_ccu_prepare_eid_log_env; printf "%s|%s|%s" "$ASCEND_GLOBAL_LOG_LEVEL" "$ASCEND_MODULE_LOG_LEVEL" "$ASCEND_SLOG_PRINT_TO_STDOUT"'
        for settings, expected in (
            ({}, '3|HCCL=3:HCCP=3|0'),
            ({'ASCEND_GLOBAL_LOG_LEVEL': '1'}, '1|HCCL=1:HCCP=1|0'),
            ({'ASCEND_GLOBAL_LOG_LEVEL': '1', 'ASCEND_MODULE_LOG_LEVEL': 'HCCL=0', 'ASCEND_SLOG_PRINT_TO_STDOUT': '1'}, '1|HCCL=0|1'),
            ({'ASCEND_GLOBAL_LOG_LEVEL': '3', 'ASCEND_MODULE_LOG_LEVEL': 'HCCL=3', 'A5_EID_LOG_LEVEL': '0', 'A5_EID_LOG_STDOUT': '1'}, '0|HCCL=0:HCCP=0|1'),
        ):
            result = subprocess.run(['bash', '-c', script, 'eid-log-test', str(ROOT/'scripts/a5_ccu_test_env.sh')],
                                    env=dict(clean, **settings), capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout, expected)

    @unittest.skipUnless(shutil.which('g++'), 'g++ required')
    def test_create_trace_preserves_failure_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'trace.cc'
            source.write_text('''#include "create_trace.h"
#include <cassert>
#include <stdexcept>
int main() {
    a5ccu::CreateTrace trace("scope=test rank=2"); int calls=0;
    assert(trace.Call("ok",[&]{++calls;return 0;})==0);
    assert(trace.Call("register_kernel",[&]{++calls;return 4;})==4);
    assert(trace.Call("driver",[&]{++calls;return 4098;})==4098);
    try { trace.Call("build",[&]()->int{++calls;throw std::runtime_error("unsupported rep");}); }
    catch(const std::runtime_error&) {assert(calls==4);return 0;}
    return 1;
}''')
            binary = str(Path(tmp) / 'trace')
            subprocess.run(['g++', '-std=c++14', '-Wall', '-Wextra', '-Werror',
                '-I'+str(ROOT/'examples/a5_ccu_urma_route_probe/common'), str(source), '-o', binary], check=True)
            result = subprocess.run([binary], capture_output=True, text=True, check=True)
            self.assertIn('stage=register_kernel status=4', result.stderr)
            self.assertIn('stage=driver status=4098', result.stderr)
            self.assertIn('stage=build error=unsupported rep', result.stderr)

    def test_create_only_is_not_a_correctness_or_performance_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'results').mkdir(); (root/'plans').mkdir()
            (root/'exit.status').write_text('0')
            (root/'plans/topology.json').write_text('{}')
            (root/'run.log').write_text(''.join(f'EP8_EID_CREATE_RANK_PASS rank={r}\n' for r in range(8)))
            for r in range(8):
                (root/'results'/f'rank{r}.json').write_text(json.dumps(dict(rank=r,
                    backend='participant_eid_ccu_urma', options=dict(suite='create'), reports=[])))
            cmd = [sys.executable, str(ROOT/'scripts/summarize_a5_ccu_eid.py'), str(root)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertIn('EP8_EID_CREATE_ALL_RANKS_AND_EXIT_PASS', result.stdout)
            self.assertNotIn('EP8_EID_ALL_RANKS_AND_EXIT_PASS', result.stdout)
            self.assertNotEqual(subprocess.run(cmd+['--compare',str(root)], capture_output=True).returncode,0)
            (root/'run.log').write_text('EP8_EID_CREATE_RANK_PASS rank=0\n')
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode,0)

    @unittest.skipUnless(shutil.which('g++'), 'g++ required')
    def test_production_ccu_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = str(Path(tmp) / 'graph')
            subprocess.run(['g++', '-std=c++14', '-Wall', '-Wextra', '-Werror',
                '-DA5_TEST_CCU_V1_ONLY',
                '-I' + str(ROOT / 'tests/cpp/fake_ccu'),
                str(ROOT / 'tests/cpp/test_a5_ccu_eid.cpp'),
                str(ROOT / 'examples/a5_ccu_urma_route_probe/op_kernel_ccu/eid_alltoallv_kernel.cc'),
                str(ROOT / 'tests/cpp/group_copy_model_stub.cc'), '-o', binary], check=True, timeout=90)
            result = subprocess.run([binary], capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('EP8_EID_GRAPH_PASS epochs=100 launches=800', result.stdout)
            self.assertIn('EP8_EID_REGISTER_STAGES_PASS stages=9 payload_disabled=8', result.stdout)
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
