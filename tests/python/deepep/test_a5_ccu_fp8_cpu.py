"""FP8 byte transport, including NaNs, signed zero, guards and relay slots."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python/deep_ep/deep_ep'))
from ccu_payload import fp8_dtypes, byte_payload_view, copy_payload_
from a5_ccu_payload_test_utils import (make_payload, full_payload, shifted_payload,
                                     equal_payload, guards_intact)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch required')
class Fp8Tests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'), 'g++ required')
    def test_production_cpp_copy_and_pipeline_dtype(self):
        import torch
        torch_root = Path(torch.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as folder:
            binary = str(Path(folder) / 'fp8')
            subprocess.run(['g++', '-std=c++17', '-O1',
                '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch.compiled_with_cxx11_abi())),
                '-I' + str(torch_root / 'include'),
                '-I' + str(torch_root / 'include/torch/csrc/api/include'),
                str(ROOT / 'tests/cpp/test_a5_ccu_fp8_payload.cpp'),
                '-L' + str(torch_root / 'lib'), '-Wl,-rpath,' + str(torch_root / 'lib'),
                '-ltorch_cpu', '-lc10', '-o', binary], check=True, timeout=180)
            result = subprocess.run([binary], capture_output=True, text=True, check=True, timeout=60)
            self.assertIn('FP8_CPP_PAYLOAD_PASS formats=4 pipeline_runs=400', result.stdout)

    def test_all_byte_patterns_views_and_guarded_copy(self):
        import torch
        self.assertGreaterEqual(len(fp8_dtypes()), 2)
        for dtype in fp8_dtypes():
            with self.subTest(dtype=dtype):
                raw = torch.arange(256, dtype=torch.int64).to(torch.uint8)
                payload = raw.view(dtype)
                self.assertEqual(byte_payload_view(payload).data_ptr(), raw.data_ptr())
                store = full_payload((256 + 74,), -17, dtype, 'cpu')
                target = store[37:-37]
                copy_payload_(target, payload)
                self.assertTrue(equal_payload(payload, target))
                self.assertTrue(guards_intact(store, 37, -17))
                self.assertEqual(target.storage_offset(), 37)
                self.assertTrue(torch.equal(byte_payload_view(shifted_payload(payload, 1)),
                                            ((raw.to(torch.int16) + 1) % 256).to(torch.uint8)))

    def test_python_pipeline_all_ranks_reuses_slots_and_changes_counts(self):
        import torch
        import test_a5_ccu_pipeline_cpu as ref
        from heuristic_alltoallv import make_plan
        from pipelined_alltoallv import PipelineContext, PipelineLane, PreparedPipelinedAllToAllV

        class ByteRuntime(ref.FakeRuntime):
            def copy(self, dst, di, src, si, n):
                self.engine.append(self.active, ('copy', self.active),
                    lambda: copy_payload_(dst.narrow(0, di, n), src.narrow(0, si, n)))
            def transfer(self, lane, send, recv, counts, storage_counts=None):
                super().transfer(lane, byte_payload_view(send), byte_payload_view(recv), counts, storage_counts)

        for dtype in fp8_dtypes():
            engine = ref.Engine()
            completed = set()
            runtimes = [ByteRuntime(engine, r) for r in range(8)]
            contexts = [PipelineContext([PipelineLane(ref.Buffer(r, l), l + 1, (r, l))
                        for l in range(3)], _runtime=runtimes[r]) for r in range(8)]
            try:
                for epoch in range(12):
                    counts = [0] * 64
                    if epoch % 4 == 1:
                        for r in range(8): counts[r * 8 + r] = r + 1
                    elif epoch % 4 == 2:
                        counts[4], counts[0] = 65, 3
                    elif epoch % 4 == 3:
                        counts = [(s * 7 + d * 11 + epoch) % 9 if s != 7 and d != 6 else 0
                                  for s in range(8) for d in range(8)]
                    plan = make_plan(counts, 257, ref.ALL)
                    sends = [make_payload(sum(counts[r*8:r*8+8]), 257, r, dtype, epoch) for r in range(8)]
                    oracle = ref.collective([byte_payload_view(t).tolist() for t in sends], counts)
                    stores = [full_payload((len(v) * 257 + 74,), -17, dtype, 'cpu') for v in oracle]
                    outputs = [store[37:-37].reshape(len(v), 257) for store, v in zip(stores, oracle)]
                    prepared = [PreparedPipelinedAllToAllV(contexts[r], plan, sends[r], chunk_bytes=7*257)
                                for r in range(8)]
                    for r in range(8): prepared[r].execute(sends[r], outputs[r])
                    pending = set(engine.nodes) - completed
                    while pending:
                        ready = [node for node in pending if node.deps <= completed]
                        self.assertTrue(ready, 'deadlocked FP8 pipeline')
                        relay = [node for node in ready if node.tag[:2] != ('collective', 0)]
                        node = (relay or ready)[0]
                        node.action(); pending.remove(node); completed.add(node)
                    for r in range(8):
                        expected = torch.tensor(oracle[r], dtype=torch.uint8).reshape(-1, 257)
                        self.assertTrue(torch.equal(byte_payload_view(outputs[r]), expected))
                        self.assertTrue(guards_intact(stores[r], 37, -17))
            finally:
                contexts.clear()  # CPU fake streams own no HCCL groups/handles


if __name__ == '__main__':
    unittest.main()
