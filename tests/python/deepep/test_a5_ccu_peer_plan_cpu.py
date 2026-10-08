#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_a5_ccu_peer_plan import build_rows, validate_policy
from analyze_a5_ccu_peer_plan import analyze


class PeerPlanTests(unittest.TestCase):
    def test_direct(self):
        for devices in ([2, 3], [0, 1, 2, 3], [3, 2, 1, 0]):
            rows, data = build_rows(devices, list(range(8)), {}, [], [])
            k = len(devices)
            self.assertEqual(len(rows), k * (k - 1) // 2)
            self.assertEqual(data["channels_per_rank"], [k - 1] * k)
            self.assertEqual(data["relay_limit_per_peer"], 6 if k == 2 else 1)

    def test_policy(self):
        valid = {"0,1": [4], "2,3": [4], "0,2": [5], "1,3": [5], "0,3": [6], "1,2": [6]}
        validate_policy([0, 1, 2, 3], list(range(8)), valid)
        for count in (1, 3, 5, 6):
            validate_policy([2, 3], list(range(8)), {"2,3": [0, 1, 4, 5, 6, 7][:count]})
        for policy in ({"0,1": [0]}, {"0,1": [4, 5]}, {"0,1": [4], "0,2": [4]}, {"1,0": [4]}):
            with self.assertRaises(ValueError):
                validate_policy([0, 1, 2, 3], list(range(8)), policy)

    def test_resolve_edge_join(self):
        entries = [dict(device=d, die=0, port_key=p, name="udmac0d0e3", index=0,
                        eid=f"0000:0000:{p:04x}:0300:0010:0000:df12:{d:04x}")
                   for d, p in ((2, 6), (3, 3))]
        edges = [dict(net_layer=0, link_type="PEER2PEER", protocols=["UB_CTP"],
                      local_a=a, local_b=4, local_a_ports=[port], local_b_ports=[relayport])
                 for a, port, relayport in ((2, "0/6", "1/0"), (3, "0/3", "1/3"))]
        rows, data = build_rows([2, 3], list(range(8)), {"2,3": [4]}, entries, edges)
        self.assertEqual(rows[1]["src_eid"], entries[0]["eid"])
        self.assertEqual(rows[1]["dst_eid"], entries[1]["eid"])
        self.assertEqual(data["relay_edges"][0]["relay_die"], 1)
        self.assertEqual(data["channels_per_rank"], [2, 2])

    def test_incomplete_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "plans").mkdir(); (root / "cases").mkdir()
            (root / "plans/peer_plan.json").write_text(json.dumps(dict(devices=[0, 1, 2, 3], relay_limit_per_peer=1)))
            log = root / "cases/peer_plan_r1.log"
            log.with_suffix(".status").write_text("0")
            results = [dict(rank=r, ranks=4, correctness="PASS", host_call_us=[10], iterations=1, host_avg_us=10)
                       for r in range(4)]
            log.write_text("\n".join("RESULT_JSON " + json.dumps(r) for r in results[:3]))
            self.assertFalse(analyze(root)["complete"])
            log.write_text("\n".join("RESULT_JSON " + json.dumps(r) for r in results))
            self.assertTrue(analyze(root)["complete"])
            (root / "run_settings.json").write_text('{"repeats":2}')
            self.assertFalse(analyze(root)["complete"])
            (root / "run_settings.json").unlink()
            log.with_suffix(".status").unlink()
            self.assertFalse(analyze(root)["complete"])


if __name__ == "__main__":
    unittest.main()
