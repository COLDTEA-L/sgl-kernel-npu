#!/usr/bin/env python3
import unittest

from a5_h2d_io_test_support import (
    bandwidth_gbps,
    direction_for,
    interval_overlap,
    parse_cases,
    parse_positive_csv,
)


class H2dIoSupportTest(unittest.TestCase):
    def test_parse_sizes(self):
        self.assertEqual(parse_positive_csv("64,1024,4096", "--sizes"), [64, 1024, 4096])
        for text in ("", "1,,2", "0", "1,-2", "1,1"):
            with self.assertRaises(ValueError):
                parse_positive_csv(text, "--sizes")

    def test_cases_and_directions(self):
        self.assertEqual(parse_cases("h2d_solo_rank0,d2h_both"), ["h2d_solo_rank0", "d2h_both"])
        self.assertEqual(direction_for("h2d_solo_rank0", 0), "h2d")
        self.assertEqual(direction_for("h2d_solo_rank0", 1), "idle")
        self.assertEqual(direction_for("mixed_rank0_d2h_rank1_h2d", 1), "h2d")
        with self.assertRaises(ValueError):
            parse_cases("unknown")

    def test_bandwidth(self):
        self.assertAlmostEqual(bandwidth_gbps(1_000_000, 1000), 1.0)

    def test_overlap(self):
        result = interval_overlap(0, 10, 2, 8)
        self.assertEqual(result["overlap_ns"], 6)
        self.assertEqual(result["shorter_fraction"], 1.0)
        self.assertEqual(result["longer_fraction"], 0.6)
        disjoint = interval_overlap(0, 2, 3, 5)
        self.assertEqual(disjoint["overlap_ns"], 0)


if __name__ == "__main__":
    unittest.main()
