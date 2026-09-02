#!/usr/bin/env python3
"""Compatibility entry point; use test_a5_aiv_urma_all2all_detour.py."""

import runpy
from pathlib import Path


target = Path(__file__).with_name("test_a5_aiv_urma_all2all_detour.py")
print(f"NOTE: this test was renamed; forwarding to {target.name}", flush=True)
runpy.run_path(str(target), run_name="__main__")
