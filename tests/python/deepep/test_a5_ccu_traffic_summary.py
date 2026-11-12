import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from summarize_a5_ccu_eid import traffic_fields


class TrafficSummaryTests(unittest.TestCase):
    def test_l1_uniform_reroutes_without_duplicating_source_payload(self):
        result = traffic_fields(dict(relay_bytes=32704*5120,
            original_network_bytes=56*3072*5120, transmitted_network_bytes=56*3072*5120,
            l1_rows=32704, participant_relay_rows=0))
        self.assertIn('moved_bytes=167444480', result)
        self.assertIn('extra_network_bytes=0 ', result)
        self.assertNotIn('extra_bytes=', result)

    def test_mixed_only_participant_hop_adds_network_bytes(self):
        result = traffic_fields(dict(relay_bytes=500*5120,
            original_network_bytes=1000*5120, transmitted_network_bytes=1100*5120,
            l1_rows=400, participant_relay_rows=100))
        self.assertIn('moved_bytes=2560000', result)
        self.assertIn('extra_network_bytes=512000', result)

    def test_old_incomplete_report_does_not_invent_extra_traffic(self):
        self.assertIn('extra_network_bytes=unknown', traffic_fields(dict(relay_bytes=50)))

    def test_invalid_accounting_is_rejected(self):
        with self.assertRaises(ValueError):
            traffic_fields(dict(relay_bytes=10, original_network_bytes=100, transmitted_network_bytes=90))


if __name__ == '__main__': unittest.main()
