"""Process-start L1 gate, shared by topology compilation and EP8 planning.

Unset preserves explicit-config opt-in for existing deployments. 0 overrides
that opt-in. This is not a hot switch for already registered handles/plans.
"""
import os


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
    return config if enabled else None


def planner_l1_options(strategy, resources, budget):
    # Validate even for L0 policies, but enable does not replace an explicitly
    # chosen greedy/bottleneck strategy with L1.
    enabled = l1_enabled(strategy == 'l1_bottleneck')
    if not enabled and os.environ.get('A5_ENABLE_L1') == '0':
        return ('bottleneck' if strategy == 'l1_bottleneck' else strategy), None, None
    return strategy, resources, budget
