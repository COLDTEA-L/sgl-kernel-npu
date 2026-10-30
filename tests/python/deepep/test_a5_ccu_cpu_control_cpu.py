import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest

from a5_ccu_cpu_control import CpuControl, launch_store_path


def worker(path, rank, mismatch, results):
    try:
        control = CpuControl(path, rank, 8, timeout_seconds=5)
        control.check_config('different' if mismatch and rank == 3 else 'same')
        control.barrier('finished')
        results.put((rank, 'pass'))
    except Exception as error:
        results.put((rank, str(error)))


class CpuControlTests(unittest.TestCase):
    def run_ranks(self, mismatch):
        # Only CPU store APIs; fork avoids loading eight copies of Torch at once.
        import torch.distributed
        context = mp.get_context('fork')
        with tempfile.TemporaryDirectory() as tmp:
            results = context.Queue()
            processes = [context.Process(target=worker,
                args=(str(Path(tmp)/'control.store'), rank, mismatch, results)) for rank in range(8)]
            try:
                for process in processes:
                    process.start()
                values = dict(results.get(timeout=15) for _ in processes)
                for process in processes:
                    process.join(timeout=5)
                    self.assertEqual(process.exitcode, 0)
                return values
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join()
                results.close()

    def test_eight_ranks_agree_and_finish_without_process_group(self):
        self.assertEqual(self.run_ranks(False), {rank: 'pass' for rank in range(8)})

    def test_every_rank_rejects_configuration_mismatch(self):
        values = self.run_ranks(True)
        self.assertEqual(len(values), 8)
        self.assertTrue(all('rank configuration mismatch' in value for value in values.values()))

    def test_missing_rank_times_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = CpuControl(Path(tmp)/'missing.store', 0, 2, timeout_seconds=0.2)
            with self.assertRaisesRegex(RuntimeError, 'CPU_CONTROL_FAILED phase=config rank=0'):
                control.check_config('same')

    def test_launch_and_restart_isolate_stale_keys(self):
        a = launch_store_path('/results', {'TORCHELASTIC_RUN_ID': 'first'})
        b = launch_store_path('/results', {'TORCHELASTIC_RUN_ID': 'second'})
        c = launch_store_path('/results', {'TORCHELASTIC_RUN_ID': 'first', 'TORCHELASTIC_RESTART_COUNT': '1'})
        self.assertEqual(len({a, b, c}), 3)
        with self.assertRaises(ValueError):
            launch_store_path('/results', {})


if __name__ == '__main__':
    unittest.main()
