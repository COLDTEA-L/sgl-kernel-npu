import os
import sys

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
opp_path = os.path.join(current_dir, "vendors", "hwcomputing")
lib_path = os.path.join(current_dir, "vendors", "hwcomputing", "op_api", "lib")
# Set environment variables related to custom operators
os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
    f"{opp_path}:{os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}"
)
os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Wheels install the extension inside the ``deep_ep`` package.  Register a
# compatibility alias before importing the remaining modules, which still use
# the historical top-level ``deep_ep_cpp`` name when running from a source tree.
try:
    from . import deep_ep_cpp as deep_ep_cpp
except ImportError:
    import deep_ep_cpp as deep_ep_cpp

sys.modules.setdefault("deep_ep_cpp", deep_ep_cpp)
Config = deep_ep_cpp.Config

# Import strategies to register them
from . import strategies
from .buffer import Buffer
from .ep_strategy import LowLatencyStrategy, NormalStrategy
from .utils import EventOverlap
