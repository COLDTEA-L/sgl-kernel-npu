"""Process-start L1 gate, shared by topology compilation and EP8 planning.

Unset preserves explicit-config opt-in for existing deployments. 0 overrides
that opt-in. This is not a hot switch for already registered handles/plans.
"""
import os
import json
from pathlib import Path


def l1_endpoint_mode():
    mode = os.environ.get("A5_L1_ENDPOINT_MODE", "rank_graph")
    if mode not in ("rank_graph", "explicit_single"):
        raise ValueError("A5_L1_ENDPOINT_MODE must be rank_graph or explicit_single")
    return mode


def validate_l1_schema(schema):
    mode = l1_endpoint_mode()
    expected = "a5.rank-graph-l1.v1" if mode == "rank_graph" else "a5.single-port-l1.v1"
    if schema != expected:
        raise ValueError("L1 endpoint mode {} requires {}; regenerate policy/catalog".format(mode, expected))
    return mode


def l1_enabled(requested=False):
    value = os.environ.get('A5_ENABLE_L1')
    if value is None:
        return bool(requested)
    if value not in ('0', '1'):
        raise ValueError('A5_ENABLE_L1 must be 0 or 1 (or unset for explicit-config opt-in)')
    return value == '1'


def select_l1_config(config):
    enabled = l1_enabled(config is not None)
    if enabled and config is None:
        raise ValueError('A5_ENABLE_L1=1 requires an explicit --l1-config / A5_L1_CONFIG')
    print('L1_SWITCH enabled={} ignored_config={}'.format(int(enabled), int(not enabled and config is not None)))
    if enabled:
        validate_l1_schema(json.loads(Path(config).read_text()).get('schema'))
    return config if enabled else None


def planner_l1_options(strategy, resources, budget):
    # Validate even for L0 policies, but enable does not replace an explicitly
    # chosen greedy/bottleneck strategy with L1.
    enabled = l1_enabled(strategy == 'l1_bottleneck')
    if not enabled and os.environ.get('A5_ENABLE_L1') == '0':
        return ('bottleneck' if strategy == 'l1_bottleneck' else strategy), None, None
    return strategy, resources, budget
