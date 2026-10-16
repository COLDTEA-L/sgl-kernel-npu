#!/usr/bin/env python3
import ast
from functools import partial
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from types import SimpleNamespace
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
    suite_phases,
    validate_contention_devices,
)


class PathContentionSupportTest(unittest.TestCase):
    def test_launcher_separates_solo_and_resident_lifetimes(self):
        repo = Path(__file__).resolve().parents[3]
        # Exercise the real shell orchestration; replace only device-worker
        # entrypoints. Inline planning and the analyzer still use real Python.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindir = root / "bin"
            bindir.mkdir()
            cann = root / "cann"
            cann.mkdir()
            (cann / "set_env.sh").write_text(":\n")
            topology = root / "topology"
            topology.touch()
            interpreter = bindir / "python3"
            interpreter.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''\
                import json, os, pathlib, sys, time
                args = sys.argv[1:]
                def value(flag): return args[args.index(flag) + 1]
                def record(text):
                    with open(os.environ['FAKE_SUITE_EVENTS'], 'a') as handle:
                        handle.write(text + '\\n')
                if args and args[0].endswith('resolve_a5_explicit_multirelay_eids.py'):
                    src, dst, relay = [int(value(flag)) for flag in ('--src-phy', '--dst-phy', '--relay-phys')]
                    forward = src < relay
                    print('route_id\\tsrc_phy\\tdst_phy\\trelay_phy\\tweight\\tsrc_die\\tsrc_port\\trelay_die\\trelay_port_from_src\\tsrc_edge')
                    print(f'{1000+relay}\\t{src}\\t{dst}\\t{relay}\\t1\\t0\\t{1 if forward else 2}\\t0\\t{2 if forward else 1}\\tedge-{min(src,relay)}-{max(src,relay)}')
                elif args and args[0] == '-m':
                    root = pathlib.Path(value('--suite-run-dir'))
                    mode, job = value('--suite-mode'), value('--suite-job')
                    record(f'start {mode} {job}')
                    other = 'Q' if job == 'P' else 'P'
                    if mode == 'solo':
                        assert not (root / f'live_{other}').exists()
                    (root / f'live_{job}').touch()
                    (root / f'ready_{mode}_{job}').touch()
                    if mode != 'solo':
                        deadline = time.monotonic() + 5
                        while not (root / f'ready_{mode}_{other}').exists():
                            assert time.monotonic() < deadline
                            time.sleep(.01)
                    sys.path.insert(0, os.environ['FAKE_SUITE_SUPPORT'])
                    from a5_ccu_path_contention_support import format_size_label, suite_phases
                    iters = int(value('--iters'))
                    for size in map(int, value('--suite-sizes').split(',')):
                        label = format_size_label(size)
                        for repeat in range(1, int(value('--suite-repeats')) + 1):
                            for phase in suite_phases(mode, job):
                                if phase not in (f'solo_baseline_{job}', f'baseline_{job}', 'concurrent_PQ'): continue
                                case = pathlib.Path(phase) / job if phase == 'concurrent_PQ' else pathlib.Path(phase)
                                trace = root / 'traces' / f'size_{label}' / f'repeat_{repeat}' / case
                                status = root / 'status' / f'size_{label}' / f'repeat_{repeat}' / case
                                trace.mkdir(parents=True, exist_ok=True)
                                status.mkdir(parents=True, exist_ok=True)
                                for rank in (0, 1):
                                    payload = dict(rank=rank, iterations=iters, completion_mode='batch',
                                        iteration_timing_semantics='batch_only', loop_body='launch_only',
                                        batch_started_ns=1000, batch_ended_ns=10000, batch_average_us=4,
                                        records=[], pid=os.getpid())
                                    (trace / f'rank{rank}.json').write_text(json.dumps(payload))
                                    (status / f'rank{rank}.json').write_text(json.dumps(dict(host_batch_avg_us=4)))
                            if mode != 'solo':
                                for phase in suite_phases(mode, job):
                                    for expected_job in ('P', 'Q'):
                                        case = pathlib.Path(phase) / expected_job if phase == 'concurrent_PQ' else pathlib.Path(phase)
                                        if phase != 'concurrent_PQ' and phase != f'baseline_{expected_job}': continue
                                        deadline = time.monotonic() + 5
                                        path = root / 'status' / f'size_{label}' / f'repeat_{repeat}' / case / 'rank1.json'
                                        while not path.exists():
                                            assert time.monotonic() < deadline
                                            time.sleep(.01)
                                progress = root / 'progress' / mode
                                progress.mkdir(parents=True, exist_ok=True)
                                (progress / f'size_{label}_repeat_{repeat}.json').write_text('{}')
                    for rank in (0, 1):
                        print('CASE_DATA_PLANE_POLICY self_copy=ccu channel_sync=baseline')
                        print('CCU_LOCAL_REMOTE_PLAN self_copy=GroupCopy8x32K')
                        print('CASE_PREPARED_PLAN mock=1')
                    (root / f'live_{job}').unlink()
                    record(f'end {mode} {job}')
                    print(f'PASS: persistent_contention_suite job={job} mock=1')
                elif '--runtime-check' in args:
                    pass
                else:
                    os.execv(os.environ['FAKE_SUITE_PYTHON'], [os.environ['FAKE_SUITE_PYTHON']] + args)
            '''))
            interpreter.chmod(0o755)
            timeout = bindir / "timeout"
            timeout.write_text(f"#!{sys.executable}\nimport os, sys\nos.execvp(sys.argv[4], sys.argv[4:])\n")
            timeout.chmod(0o755)
            events = root / "events"
            env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}",
                       FAKE_SUITE_EVENTS=str(events), FAKE_SUITE_PYTHON=sys.executable,
                       FAKE_SUITE_SUPPORT=str(repo / "tests/python/deepep"))
            for experiment in ("baseline-check", "contention"):
                events.write_text("")
                result = subprocess.run(["bash", str(repo / "scripts/run_a5_ccu_multipath_contention.sh"),
                    "--experiment", experiment, "--p-devices", "2,3", "--p-relay", "6",
                    "--q-devices", "6,7", "--q-relay", "2", "--sizes", "256", "--warmup", "1",
                    "--iters", "2", "--no-mindstudio-profile", "--cann-root", str(cann),
                    "--topology", str(topology), "--topology-json", str(topology),
                    "--output-root", str(root / experiment)], cwd=repo, env=env,
                    text=True, capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                lifecycle = events.read_text().splitlines()
                if experiment == "baseline-check":
                    self.assertEqual(lifecycle[:4], ["start solo P", "end solo P", "start solo Q", "end solo Q"])
                    self.assertEqual(set(lifecycle[4:]), {"start resident P", "end resident P", "start resident Q", "end resident Q"})
                    self.assertIn("BASELINE_COMPLETE mode=solo job=P", result.stdout)
                else:
                    self.assertEqual(set(lifecycle), {"start contention P", "end contention P", "start contention Q", "end contention Q"})
                self.assertIn("SIZE_COMPLETE size=256B", result.stdout)

    def test_suite_isolation(self):
        self.assertEqual(suite_phases("solo", "P"), ("solo_baseline_P",))
        self.assertEqual(suite_phases("solo", "Q"), ("solo_baseline_Q",))
        self.assertEqual(suite_phases("resident", "P"), ("baseline_P", "baseline_Q"))
        self.assertNotIn("concurrent_PQ", suite_phases("resident", "Q"))
        self.assertEqual(suite_phases("contention", "P")[-1], "concurrent_PQ")
        with self.assertRaises(ValueError):
            suite_phases("invalid", "P")

    def test_solo_worker_profiles_only_plain_launches(self):
        # Execute the real suite with a fake device, without importing its
        # executable NPU bootstrap. This checks output/gate/phase behavior.
        source = (Path(__file__).with_name("test_a5_ccu_urma_multiroute_all2all.py")).read_text()
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_persistent_contention_suite")
        events = []
        barriers = []
        class Tensor:
            def __init__(self, shape):
                self.rows = [[0] * shape[1] for _ in range(shape[0])]
            def __getitem__(self, row):
                return SimpleNamespace(fill_=lambda value: self.rows[row].__setitem__(slice(None), [value] * len(self.rows[row])))
            def fill_(self, value):
                for row in range(len(self.rows)):
                    self[row].fill_(value)
            def copy_(self, other):
                self.rows = [row[:] for row in other.rows]
        class Profiler:
            def start(self): events.append("profile_start")
            def stop(self): events.append("profile_stop")
        class Buffer:
            def ccu_urma_prepared_multipath_alltoall_policy_out(self, send, recv, handle, weights):
                events.append("launch")
                epoch = send.rows[0][0] - 1
                for src in (0, 1):
                    recv[src].fill_(src * 100 + 1 + epoch)
        def assert_close(output, expected, **unused):
            self.assertEqual(output.rows, expected.rows)
        def validate(prepare, launch, snapshot, synchronize, check, count):
            for index in range(count):
                prepare(index)
                launch()
                snapshot(index)
            synchronize()
            for index in range(count): check(index)
        torch = SimpleNamespace(float32="float32",
            empty=lambda shape=None, **unused: Tensor(shape) if shape else SimpleNamespace(element_size=lambda: 4),
            empty_like=lambda tensor: Tensor((len(tensor.rows), len(tensor.rows[0]))),
            testing=SimpleNamespace(assert_close=assert_close),
            npu=SimpleNamespace(synchronize=lambda: events.append("sync"), empty_cache=lambda: None))
        namespace = dict(Path=Path, torch=torch, partial=partial, time=time, os=__import__("os"),
            json=json, parse_sizes=parse_sizes, suite_phases=suite_phases,
            safe_name=safe_name, format_size_label=format_size_label,
            mark_phase=lambda phase: None, validate_queued_calls=validate,
            make_profiler=lambda *args: Profiler(),
            file_barrier=lambda *args: barriers.append(args),
            contention_file_barrier=lambda *args: self.fail("solo invoked four-rank barrier"),
            measurement_start=lambda *args: 0,
            write_iteration_trace=__import__("a5_ccu_path_contention_support").write_iteration_trace)
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, "suite_under_test", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(suite_run_dir=directory, suite_sizes="256", suite_job="P",
                suite_mode="solo", dtype="float32", suite_repeats=1, warmup=2, iters=5,
                profile=True, implementation="prepared", self_copy_mode="ccu",
                channel_sync_mode="baseline", completion_mode="batch")
            namespace[function.name](args, Buffer(), 1, 0, 2, [1, 1])
            begin, end = events.index("profile_start"), events.index("profile_stop")
            self.assertEqual(events[begin + 1:end], ["sync"] + ["launch"] * 5 + ["sync"])
            self.assertEqual(events[:begin].count("launch"), 4)  # two checks + warmup2
            trace = json.loads((Path(directory) / "traces/size_256B/repeat_1/solo_baseline_P/rank0.json").read_text())
            self.assertEqual(trace["records"], [])
            self.assertEqual(trace["iterations"], 5)
            self.assertEqual(trace["loop_body"], "launch_only")
            self.assertTrue(barriers)

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

    def test_parameterized_contention_devices(self):
        self.assertEqual(
            validate_contention_devices("2,3", "6", "6,7", "2"),
            ((2, 3), 6, (6, 7), 2),
        )
        self.assertEqual(
            validate_contention_devices("0,1", "4", "4,5", "0"),
            ((0, 1), 4, (4, 5), 0),
        )
        for values in (
            ("2,2", "6", "6,7", "2"),
            ("2,3", "6", "3,7", "2"),
            ("2,3", "5", "6,7", "2"),
            ("2,3", "6", "6,7", "5"),
            ("2,3", "7", "6,7", "2"),
        ):
            with self.assertRaises(ValueError):
                validate_contention_devices(*values)

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
        self.assertNotIn('intentionally pinned', runner)
        self.assertIn('resolve_manifest "$p_src" "$p_dst"', runner)
        self.assertIn('resolve_manifest "$q_src" "$q_dst"', runner)

    def test_failure_diagnostics_preserve_the_first_hccl_status(self):
        root = Path(__file__).resolve().parents[3]
        exception_header = (root / "csrc/deepep/exception.hpp").read_text()
        launch_source = (root /
            "examples/a5_ccu_urma_route_probe/op_host/all_to_all_multiroute.cc").read_text()
        worker_source = (root /
            "tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py").read_text()

        self.assertIn("const HcclResult hccl_check_status = (ret);", exception_header)
        self.assertIn("std::to_string(static_cast<int>(hccl_check_status))", exception_header)
        for step in ("main_to_route_record", "route_wait_main", "ccu_launch",
                     "main_wait_route", "route_to_main_record"):
            self.assertIn(f'"{step}"', launch_source)
        self.assertIn("CASE_EXECUTE_ERROR", worker_source)
        self.assertIn('report_failure(error, phase, "warmup", index + 1, repeat)', worker_source)
        self.assertIn('report_failure(error, phase, "measure", index + 1, repeat)', worker_source)


if __name__ == "__main__":
    unittest.main()
