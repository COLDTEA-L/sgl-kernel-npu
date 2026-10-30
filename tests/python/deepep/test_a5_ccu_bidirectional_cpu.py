"""Compile and execute the production C++ host with deferred eight-rank transport."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]

@unittest.skipUnless(importlib.util.find_spec('torch') and shutil.which('g++'), 'Torch and g++ required')
class BidirectionalTests(unittest.TestCase):
    def test_production_cpp_alternating_payloads(self):
        import torch
        torch_root = Path(torch.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as folder:
            binary = str(Path(folder) / 'bidirectional')
            subprocess.run(['g++', '-std=c++17', '-O1',
                '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch.compiled_with_cxx11_abi())),
                '-I' + str(torch_root / 'include'),
                '-I' + str(torch_root / 'include/torch/csrc/api/include'),
                str(ROOT / 'tests/cpp/test_a5_ccu_bidirectional.cpp'),
                '-L' + str(torch_root / 'lib'), '-Wl,-rpath,' + str(torch_root / 'lib'),
                '-ltorch_cpu', '-lc10', '-o', binary], check=True, timeout=180)
            result = subprocess.run([binary], capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('BIDIRECTIONAL_CPP_PASS', result.stdout)
            print(result.stdout.strip())

if __name__ == '__main__':
    unittest.main()
