"""Shared utilities for radios test suite."""


def print_test_db_location():
    """Print the test database path. Call once from setUpClass."""
    try:
        from django.db import connection
        db_name = connection.settings_dict.get("NAME", "(unknown)")
    except Exception:
        db_name = "(unavailable)"
    print(f"\n  Test database: {db_name}")


def fmt_time(seconds):
    """Format a timestamp in seconds as H:MM:SS or MM:SS."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def fmt_dur(seconds):
    """Format a duration in seconds as 'Xm Ys' or 'Xs'."""
    total = int(seconds)
    m, s = divmod(total, 60)
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
