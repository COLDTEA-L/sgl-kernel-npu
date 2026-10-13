"""Import-safe helpers for the two-card H2D/D2H contention experiment."""

CASE_DIRECTIONS = {
    "h2d_solo_rank0": ("h2d", "idle"),
    "h2d_solo_rank1": ("idle", "h2d"),
    "h2d_both": ("h2d", "h2d"),
    "d2h_solo_rank0": ("d2h", "idle"),
    "d2h_solo_rank1": ("idle", "d2h"),
    "d2h_both": ("d2h", "d2h"),
    "mixed_rank0_h2d_rank1_d2h": ("h2d", "d2h"),
    "mixed_rank0_d2h_rank1_h2d": ("d2h", "h2d"),
}

DEFAULT_CASES = tuple(CASE_DIRECTIONS)


def parse_positive_csv(text, name):
    parts = [item.strip() for item in text.split(",")]
    if not parts or any(not item or not item.isdecimal() for item in parts):
        raise ValueError(f"{name} must be a comma-separated positive integer list")
    values = [int(item) for item in parts]
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain only positive integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def parse_cases(text):
    names = [item.strip() for item in text.split(",")]
    if not names or any(not item for item in names):
        raise ValueError("--cases must be a non-empty comma-separated list")
    unknown = [name for name in names if name not in CASE_DIRECTIONS]
    if unknown:
        raise ValueError(f"unknown case(s): {','.join(unknown)}")
    if len(names) != len(set(names)):
        raise ValueError("--cases must not contain duplicates")
    return names


def direction_for(case_name, rank):
    if rank not in (0, 1):
        raise ValueError("rank must be 0 or 1")
    try:
        return CASE_DIRECTIONS[case_name][rank]
    except KeyError as exc:
        raise ValueError(f"unknown case: {case_name}") from exc


def bandwidth_gbps(byte_count, average_us):
    if byte_count <= 0 or average_us <= 0:
        raise ValueError("positive byte count and duration required")
    return byte_count / (average_us * 1000.0)


def interval_overlap(left_start, left_end, right_start, right_end):
    if not left_start <= left_end or not right_start <= right_end:
        raise ValueError("invalid interval")
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
    left_duration = left_end - left_start
    right_duration = right_end - right_start
    shorter = min(left_duration, right_duration)
    longer = max(left_duration, right_duration)
    return {
        "overlap_ns": overlap,
        "shorter_fraction": overlap / shorter if shorter else 0.0,
        "longer_fraction": overlap / longer if longer else 0.0,
    }
