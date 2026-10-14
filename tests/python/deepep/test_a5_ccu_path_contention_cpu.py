#!/usr/bin/env python3
import tempfile
import threading
import unittest
from pathlib import Path

from a5_ccu_path_contention_support import (
    DEFAULT_SIZES,
    contention_file_barrier,
    contention_measurement_start,
    format_size_label,
    overlap_ns,
    parse_sizes,
    resolve_hccl_buffer_mb,
    safe_name,
)


class PathContentionSupportTest(unittest.TestCase):
    def test_default_size_range(self):
        self.assertEqual(DEFAULT_SIZES[0], 16 * 1024)
        self.assertEqual(DEFAULT_SIZES[-1], 256 * 1024 * 1024)
        self.assertEqual(parse_sizes("16384,65536"), [16384, 65536])
        for text in ("", "0", "1", "256,256", "256,x"):
            with self.assertRaises(ValueError):
                parse_sizes(text)
        self.assertEqual(format_size_label(16 * 1024), "16KB")
        self.assertEqual(format_size_label(256 * 1024), "256KB")
        self.assertEqual(format_size_label(1024 * 1024), "1MB")
        self.assertEqual(format_size_label(256 * 1024 * 1024), "256MB")
        self.assertEqual(format_size_label(768), "768B")

    def test_hccl_buffer_budget(self):
        self.assertEqual(resolve_hccl_buffer_mb(16 * 1024), 200)
        self.assertEqual(resolve_hccl_buffer_mb(64 * 1024 * 1024), 200)
        self.assertEqual(resolve_hccl_buffer_mb(256 * 1024 * 1024), 576)
        self.assertEqual(resolve_hccl_buffer_mb(256 * 1024 * 1024, "768"), 768)
        for setting in ("", "0", "bad"):
            with self.assertRaises(ValueError):
                resolve_hccl_buffer_mb(16 * 1024, setting)

    def test_names_and_overlap(self):
        self.assertEqual(safe_name("P"), "P")
        with self.assertRaises(ValueError):
            safe_name("P/Q")
        self.assertEqual(overlap_ns((0, 10), (4, 12)), 6)
        self.assertEqual(overlap_ns((0, 3), (3, 8)), 0)

    def test_four_rank_release(self):
        with tempfile.TemporaryDirectory() as directory:
            values = {}
            failures = []

            def participant(job, rank):
                try:
                    values[(job, rank)] = contention_measurement_start(
                        directory, job, rank, lead_seconds=0.01, timeout=2
                    )
                except Exception as error:  # pragma: no cover - failure diagnostic
                    failures.append(error)

            threads = [threading.Thread(target=participant, args=(job, rank))
                       for job in ("P", "Q") for rank in (0, 1)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertFalse(failures)
            self.assertEqual(len(values), 4)
            self.assertTrue(all(value >= 0 for value in values.values()))

    def test_four_rank_persistent_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            passed = []
            failures = []

            def participant(job, rank):
                try:
                    contention_file_barrier(
                        directory, "baseline_P_done", job, rank, timeout=2
                    )
                    passed.append((job, rank))
                except Exception as error:  # pragma: no cover - failure diagnostic
                    failures.append(error)

            threads = [threading.Thread(target=participant, args=(job, rank))
                       for job in ("P", "Q") for rank in (0, 1)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertFalse(failures)
            self.assertEqual(set(passed), {("P", 0), ("P", 1), ("Q", 0), ("Q", 1)})

    def test_runner_pins_full_ccu_baseline_policy(self):
        runner = (Path(__file__).resolve().parents[3] /
                  "scripts/run_a5_ccu_multipath_contention.sh").read_text()
        self.assertIn('self_copy_mode=ccu', runner)
        self.assertIn('channel_sync_mode=baseline', runner)
        self.assertIn('export A5_CCU_SELF_COPY_MODE="$self_copy_mode"', runner)
        self.assertIn('export A5_CCU_CHANNEL_SYNC_MODE="$channel_sync_mode"', runner)
        self.assertIn('--self-copy-mode "$self_copy_mode"', runner)
        self.assertIn('--channel-sync-mode "$channel_sync_mode"', runner)
        self.assertIn('self_copy=GroupCopy8x32K', runner)
        self.assertIn("grep -q '^SELF_COPY_LANE '", runner)
        self.assertIn('--suite-sizes "$sizes"', runner)
        self.assertIn('--plan-id "path-contention-${job}-persistent"', runner)
        self.assertIn('SIZE_COMPLETE size=', runner)


if __name__ == "__main__":
    unittest.main()
