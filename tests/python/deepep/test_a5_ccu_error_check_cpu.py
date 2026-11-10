#!/usr/bin/env python3
"""Compile/run the real bridge error macros: a failed API must never be retried."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class ErrorCheckTests(unittest.TestCase):
    def test_failed_calls_are_evaluated_once_and_keep_original_status(self):
        compiler = shutil.which('c++') or shutil.which('g++')
        if not compiler: self.skipTest('C++ compiler unavailable')
        source = r'''
#define HCCL_SUCCESS 0
#define ACL_SUCCESS 0
#include "exception.hpp"
#include <cassert>
int calls = 0;
int fail_then_succeed() { return ++calls == 1 ? 6 : 0; }
int main() {
    for (int kind = 0; kind < 2; ++kind) {
        calls = 0;
        bool caught = false;
        try {
            if (kind == 0) { HCCL_CHECK(fail_then_succeed()); }
            else { ACL_CHECK(fail_then_succeed()); }
        } catch (const deep_ep::EPException &exc) {
            caught = true;
            assert(std::string(exc.what()).find("'6'") != std::string::npos);
        }
        assert(caught && calls == 1);
        // Success also evaluates the argument once.
        if (kind == 0) { HCCL_CHECK(fail_then_succeed()); }
        else { ACL_CHECK(fail_then_succeed()); }
        assert(calls == 2);
    }
}
'''
        with tempfile.TemporaryDirectory(prefix='a5-error-check-') as temp:
            src, exe = Path(temp) / 'main.cc', Path(temp) / 'test'
            src.write_text(source)
            subprocess.run([compiler, '-std=c++17', '-I', str(ROOT / 'csrc/deepep'),
                            str(src), '-o', str(exe)], check=True, capture_output=True, text=True)
            subprocess.run([str(exe)], check=True)


if __name__ == '__main__': unittest.main()
