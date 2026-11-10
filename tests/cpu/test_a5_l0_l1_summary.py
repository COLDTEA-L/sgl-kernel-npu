import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('probe', Path(__file__).resolve().parents[2] / 'scripts/a5_ccu_l0_l1_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        opts = dict(suite='benchmark', mode='all', rounds=1, bytes=1024, l0_percent=50, timing='block')
        (self.root / 'options.json').write_text(json.dumps(opts))
        (self.root / 'exit.status').write_text('0\n')
        for rank in range(2):
            (self.root / ('rank%d.log' % rank)).write_text('L0L1_RANK_PASS rank=%d suite=benchmark[INFO]\n' % rank)
            rows = []
            for mode, left, us in [('l0', 1024, 100), ('l1', 0, 80), ('split', 512, 60)]:
                rows.append(dict(rank=rank, round=0, mode=mode, timing='block', bytes_per_direction=1024,
                                 l0_bytes=left, l1_bytes=1024-left, wall_us=us+rank*10))
            (self.root / ('rank%d.jsonl' % rank)).write_text('\n'.join(json.dumps(x) for x in rows))

    def tearDown(self):
        self.tmp.cleanup()

    def test_max_rank_not_rank0(self):
        out = probe.summarize(self.root)
        self.assertEqual(out['modes']['l0']['median_us'], 110)
        self.assertAlmostEqual(out['l0_over_split'], 110/70)

    def test_nonzero_exit_rejected(self):
        (self.root / 'exit.status').write_text('1\n')
        with self.assertRaises(ValueError): probe.summarize(self.root)

    def test_missing_samples_rejected(self):
        (self.root / 'rank1.jsonl').write_text('')
        with self.assertRaises(ValueError): probe.summarize(self.root)

    def test_wrong_split_rejected(self):
        p = self.root / 'rank1.jsonl'
        p.write_text(p.read_text().replace('"l0_bytes": 512', '"l0_bytes": 256'))
        with self.assertRaises(ValueError): probe.summarize(self.root)

    def test_discovery_does_not_require_timing(self):
        (self.root / 'options.json').write_text(json.dumps({'suite': 'discover'}))
        for rank in range(2):
            (self.root / ('rank%d.log' % rank)).write_text('L0L1_RANK_PASS rank=%d suite=discover\n' % rank)
        self.assertTrue(probe.summarize(self.root)['both_ranks_pass'])

    def test_missing_clean_marker_rejected(self):
        (self.root / 'rank1.log').write_text('')
        with self.assertRaises(ValueError): probe.summarize(self.root)


if __name__ == '__main__':
    unittest.main()
