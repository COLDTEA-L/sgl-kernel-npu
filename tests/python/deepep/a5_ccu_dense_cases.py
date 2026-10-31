"""Measured dense routing matrix plus explicitly synthetic sensitivity cases."""
OBSERVED_ROW = (3048, 3165, 3067, 3017, 3071, 3028, 3046, 3134)
DENSE_CASES = ('observed_dense', 'dense_near', 'dense_hot')


def dense_counts(name):
    # Absolute token-row counts: never multiplied by --row-scale.
    if name == 'observed_dense':
        return OBSERVED_ROW * 8
    if name not in DENSE_CASES:
        raise ValueError('unknown dense case: ' + name)
    c = [3017] * 64
    c[1] = 3165 if name == 'dense_near' else 4096
    return tuple(c)
