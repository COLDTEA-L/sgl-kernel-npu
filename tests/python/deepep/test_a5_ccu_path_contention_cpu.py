#!/usr/bin/env python3
import tempfile
import threading
import unittest

from a5_ccu_path_contention_support import (
    DEFAULT_SIZES,
    contention_measurement_start,
    overlap_ns,
    parse_sizes,
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


if __name__ == "__main__":
    unittest.main()
