"""CPU scheduling/lifetime checks for reusable counts; hardware checked by EID suite."""
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'python/deep_ep/deep_ep'))
from routing_counts import RoutingCountsExchange


class Tensor:
    def __init__(self, shape, device, dtype='int64', data=None, offset=0):
        self.shape, self.device, self.dtype, self.offset = tuple(shape), device, dtype, offset
        self.data = [0]*self.numel() if data is None else data
    def numel(self):
        n = 1
        for v in self.shape: n *= v
        return n
    def is_contiguous(self): return True
    def unbind(self, dim):
        assert dim == 0 and self.shape == (8,8)
        return [Tensor((8,), self.device, self.dtype, self.data, i*8) for i in range(8)]
    def view(self, n): return Tensor((n,), self.device, self.dtype, self.data)
    def copy_(self, other, non_blocking):
        assert non_blocking
        self.data[:] = other.data
        return self
    def tolist(self): return self.data[self.offset:self.offset+self.numel()]


class Fake:
    def __init__(self):
        self.device = SimpleNamespace(type='npu', index=0)
        self.log = []
        self.allocations = 0
        self.current = self.new_stream('producer')
        self.work_failed = False
        self.event_failed = False
        self.torch = SimpleNamespace(int64='int64', empty=self.empty, npu=SimpleNamespace(
            device=self.device_context, stream=self.stream_context, current_stream=lambda:self.current,
            Stream=lambda:self.new_stream('counts'), Event=self.event))
        self.dist = SimpleNamespace(get_world_size=lambda group:8, all_gather=self.gather)
        self.torch.distributed = self.dist
    def new_stream(self, name):
        return SimpleNamespace(name=name,
            wait_event=lambda e:self.log.append(('wait_event',name)),
            synchronize=lambda:self.log.append(('stream_sync',name)))
    @contextmanager
    def device_context(self, device): yield
    @contextmanager
    def stream_context(self, stream):
        old = self.current
        self.current = stream
        try: yield
        finally: self.current = old
    def event(self):
        def sync():
            self.log.append(('event_sync',))
            if self.event_failed: raise RuntimeError('injected D2H failure')
        return SimpleNamespace(record=lambda stream:self.log.append(('record',stream.name)), synchronize=sync)
    def empty(self, shape, device, dtype, pin_memory=False):
        self.allocations += 1
        if device == 'cpu': assert pin_memory
        return Tensor(shape, device, dtype)
    def gather(self, parts, counts, group, async_op):
        assert async_op
        self.log.append(('gather',self.current.name))
        def wait():
            self.log.append(('work_wait',self.current.name))
            if self.work_failed: raise RuntimeError('injected collective failure')
            for rank, part in enumerate(parts):
                part.data[part.offset:part.offset+8] = [v+rank*100 for v in counts.tolist()]
        return SimpleNamespace(wait=wait)
    @contextmanager
    def installed(self):
        with patch.dict(sys.modules, {'torch':self.torch,'torch.distributed':self.dist}): yield


class CountsTests(unittest.TestCase):
    def test_reuse_changing_counts_and_completion_order(self):
        for dedicated in (False, True):
            f=Fake()
            c=Tensor((8,),f.device)
            with f.installed():
                x=RoutingCountsExchange(c,dedicated_stream=dedicated)
                self.assertEqual(f.allocations,2)
                pointer=None
                for epoch in range(100):
                    c.data[:]=[epoch*1000+j for j in range(8)]
                    f.log.clear()
                    x.begin(c)
                    self.assertEqual(f.current.name,'producer')
                    self.assertIs(x._input,c)
                    out=x.finish()
                    self.assertEqual(out.tolist(),[epoch*1000+r*100+j for r in range(8) for j in range(8)])
                    if pointer is not None: self.assertIs(out,pointer)
                    pointer=out
                    self.assertIsNone(x._input)
                    labels=[v[0] for v in f.log]
                    self.assertLess(labels.index('gather'), labels.index('work_wait'))
                    self.assertEqual(labels[-1],'event_sync')
                    self.assertNotIn('stream_sync',labels)
                    if dedicated:
                        self.assertEqual(f.log[:3],[('record','producer'),('wait_event','counts'),('gather','counts')])
                self.assertEqual(f.allocations,2)
                x.close();x.close()
                with self.assertRaisesRegex(RuntimeError,'closed'):x.begin(c)
    def test_pending_close_and_misuse(self):
        f=Fake();c=Tensor((8,),f.device)
        with f.installed():
            x=RoutingCountsExchange(c)
            with self.assertRaisesRegex(RuntimeError,'before finish'):x.finish()
            with self.assertRaises(ValueError):x.begin(Tensor((64,),f.device))
            x.begin(c)
            with self.assertRaisesRegex(RuntimeError,'previous'):x.begin(c)
            x.close()
            self.assertEqual(f.log[-2:],[('work_wait','producer'),('stream_sync','producer')])
            self.assertIsNone(x._input)
    def test_failed_wait_preserves_input_and_poison(self):
        f=Fake();c=Tensor((8,),f.device)
        with f.installed():
            x=RoutingCountsExchange(c,dedicated_stream=True)
            x.begin(c);f.work_failed=True
            with self.assertRaisesRegex(RuntimeError,'collective failure'):x.finish()
            self.assertIs(x._input,c)
            with self.assertRaisesRegex(RuntimeError,'failed'):x.begin(c)
            f.work_failed=False
            x.close()
            self.assertIsNone(x._input)
    def test_failed_copy_completion_is_not_readable(self):
        f=Fake();c=Tensor((8,),f.device)
        with f.installed():
            x=RoutingCountsExchange(c)
            x.begin(c);f.event_failed=True
            with self.assertRaisesRegex(RuntimeError,'D2H failure'):x.finish()
            self.assertTrue(x._pending)
            with self.assertRaisesRegex(RuntimeError,'failed'):x.finish()
            x.close()
    def test_reuse_cannot_switch_caller_stream(self):
        f=Fake();c=Tensor((8,),f.device)
        with f.installed():
            x=RoutingCountsExchange(c)
            f.current=f.new_stream('other')
            with self.assertRaisesRegex(RuntimeError,'original caller'):x.begin(c)
            self.assertFalse(x._pending)
            x.close()

if __name__ == '__main__': unittest.main()
