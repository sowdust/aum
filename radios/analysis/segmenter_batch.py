"""
Backup of the batch segmenter before real-time streaming was introduced.

This is a verbatim copy of segmenter.py as of the batch-only pipeline.
Kept for rollback if the streaming pipeline needs to be reverted.
"""

# This file is a backup — import from segmenter.py for active use.
from radios.analysis.segmenter import *  # noqa: F401,F403
