#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_a5_ccu_peer_plan import build_rows, validate_policy, load_requested_policy
from analyze_a5_ccu_peer_plan import analyze, valid_result


class PeerPlanTests(unittest.TestCase):
    def test_command_line_relay_policy_preserves_ids_and_order(self):
        pool = [0, 1, 4, 5, 6, 7]
        for count in range(1, 7):
            relays = list(reversed(pool[:count]))
            policy = load_requested_policy([3, 2], relay_phys=",".join(map(str, relays)))
            self.assertEqual(policy, {"2,3": relays})
            validate_policy([3, 2], list(range(8)), policy)
        self.assertEqual(load_requested_policy([2, 3]), {})
        for value in ("", "0,", ",1", "0,,1", "0,0", "-1", "1.5"):
            with self.assertRaises(ValueError):
                load_requested_policy([2, 3], relay_phys=value)
        with self.assertRaises(ValueError):
            load_requested_policy([0, 1, 2, 3], relay_phys="4")
        with self.assertRaises(ValueError):
            load_requested_policy([2, 3], Path("unused.json"), "0,1")

    def test_relay_cli_validates_without_topology_or_runtime(self):
        command = [sys.executable, str(ROOT / "scripts/prepare_a5_ccu_peer_plan.py"),
                   "--devices", "3,2", "--root-info", "/not-present/rootinfo.json",
                   "--topology-json", "/not-present/topology.json", "--validate-only"]
        for relays in ("0", "0,1", "0,1,4", "0,1,4,5", "0,1,4,5,6", "0,1,4,5,6,7"):
            result = subprocess.run(command + ["--relay-phys", relays],
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout.split(" ", 1)[1])
            self.assertEqual(data["relay_policy"], {"2,3": list(map(int, relays.split(",")))})
        for options in (["--relay-phys", "2"], ["--relay-phys", "9"],
                        ["--relay-phys", "0,0"], ["--relay-phys", ""],
                        ["--relay-phys", "0", "--relay-map", "unused.json"]):
            result = subprocess.run(command + options, text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stderr)

    def test_launcher_rejects_relay_errors_before_loading_runtime(self):
        script = str(ROOT / "scripts/run_a5_ccu_peer_plan_alltoall.sh")
        for options in (["--devices", "2,3", "--relay-phys", "2"],
                        ["--devices", "2,3", "--relay-phys", "0,0"],
                        ["--devices", "0,1,2,3", "--relay-phys", "4"],
                        ["--devices", "2,3", "--relay-phys", "0", "--relay-map", "unused.json"]):
            result = subprocess.run(["bash", script, "--cann-root", "/not-present/cann"] + options,
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertNotIn("runtime_bootstrap", result.stdout)
            self.assertNotIn("Result directory:", result.stdout)

    def test_malformed_timing_results_fail_closed(self):
        valid = dict(rank=0, ranks=4, correctness="PASS", host_call_us=[10], iterations=1, host_avg_us=10)
        self.assertTrue(valid_result(valid, 4))
        for changes in (dict(rank=[]), dict(host_call_us=[float("nan")]),
                        dict(host_call_us="10"), dict(host_call_us=[-1]),
                        dict(host_avg_us=99), dict(iterations=True)):
            self.assertFalse(valid_result(dict(valid, **changes), 4))
    def test_direct(self):
        for devices in ([2, 3], [0, 1, 2, 3], [3, 2, 1, 0]):
            rows, data = build_rows(devices, list(range(8)), {}, [], [])
            k = len(devices)
            self.assertEqual(len(rows), k * (k - 1) // 2)
            self.assertEqual(data["channels_per_rank"], [k - 1] * k)
            self.assertEqual(data["relay_limit_per_peer"], 6 if k == 2 else 1)

    def test_batch_analysis_requires_gate_and_atomic_rank_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "plans").mkdir(); (root / "cases").mkdir()
            (root / "plans/peer_plan.json").write_text(json.dumps(dict(devices=[0, 1, 2, 3], relay_limit_per_peer=1)))
            (root / "run_settings.json").write_text(json.dumps(dict(repeats=1, sync_mode="batch", iterations=2)))
            (root / "cases/peer_plan_r1.status").write_text("0")
            (root / "cases/peer_plan_r1.log").write_text("RESULT_JSON {RESULT_JSON {broken\n")
            result_dir = root / "r1/results"; result_dir.mkdir(parents=True)
            for rank in range(4):
                item = dict(rank=rank, ranks=4, correctness="PASS", iterations=2, sync_mode="batch",
                    host_call_us=[], host_avg_us=10, host_batch_total_us=20,
                    batch_validation=dict(status="PASS", rounds=2, calls_per_round=2))
                (result_dir / f"rank{rank}.json").write_text(json.dumps(item))
            self.assertTrue(analyze(root)["complete"])
            item.pop("batch_validation")
            self.assertFalse(valid_result(item, 4))
            (result_dir / "rank3.json").write_text(json.dumps(item))
            self.assertFalse(analyze(root)["complete"])
            item["batch_validation"] = dict(status="PASS", rounds=2, calls_per_round=2)
            item["host_batch_total_us"] = 21
            self.assertFalse(valid_result(item, 4))
            item["host_batch_total_us"] = 20
            (result_dir / "rank3.json").write_text(json.dumps(item))
            settings = dict(repeats=1, sync_mode="per-call", iterations=2)
            (root / "run_settings.json").write_text(json.dumps(settings))
            self.assertFalse(analyze(root)["complete"])

    def test_policy(self):
        valid = {"0,1": [4], "2,3": [4], "0,2": [5], "1,3": [5], "0,3": [6], "1,2": [6]}
        validate_policy([0, 1, 2, 3], list(range(8)), valid)
        for count in (1, 3, 5, 6):
            validate_policy([2, 3], list(range(8)), {"2,3": [0, 1, 4, 5, 6, 7][:count]})
        for policy in ({"0,1": [0]}, {"0,1": [4, 5]}, {"0,1": [4], "0,2": [4]}, {"1,0": [4]}):
            with self.assertRaises(ValueError):
                validate_policy([0, 1, 2, 3], list(range(8)), policy)

    def test_resolve_edge_join(self):
        entries = [dict(device=d, die=die, port=p, net_layer=0, verified=True,
                        plane_id="fabric0", net_instance_id="machine", name="udmac0d0e3", index=0,
                        eid=f"0000:0000:0400:0300:0010:0000:df12:{d * 16 + p:04x}")
                   for d, die, p in ((2, 0, 6), (3, 0, 3), (4, 1, 0), (4, 1, 3))]
        edges = [dict(net_layer=0, link_type="PEER2PEER", protocols=["UB_CTP"],
                      local_a=a, local_b=4, local_a_ports=[port], local_b_ports=[relayport])
                 for a, port, relayport in ((2, "0/6", "1/0"), (3, "0/3", "1/3"))]
        rows, data = build_rows([2, 3], list(range(8)), {"2,3": [4]}, entries, edges)
        self.assertEqual(rows[1]["src_eid"], entries[0]["eid"])
        self.assertEqual(rows[1]["dst_eid"], entries[1]["eid"])
        self.assertEqual(data["relay_edges"][0]["relay_die"], 1)
        self.assertEqual(data["channels_per_rank"], [2, 2])
        cli_rows, _ = build_rows([2, 3], list(range(8)),
            load_requested_policy([2, 3], relay_phys="4"), entries, edges)
        self.assertEqual(cli_rows, rows)

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
            log.with_suffix(".status").write_text("137")
            log.write_text(log.read_text() + "\nPEER_CASE_PHASE rank=0 phase=destroy_process_group_begin\n")
            summary = analyze(root)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["cases"][0]["correctness"], "PASS")
            self.assertEqual(summary["cases"][0]["lifecycle"], "FAILED_OR_INTERRUPTED_AFTER_DATA_PASS")
            self.assertEqual(json.loads(summary["cases"][0]["last_phase_by_rank"])["0"],
                             "destroy_process_group_begin")
            log.with_suffix(".status").write_text("0")
            (root / "run_settings.json").write_text('{"repeats":2}')
            self.assertFalse(analyze(root)["complete"])
            (root / "run_settings.json").unlink()
            log.with_suffix(".status").unlink()
            self.assertFalse(analyze(root)["complete"])


if __name__ == "__main__":
    unittest.main()
