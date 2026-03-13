"""
Backup of the chunk-based recorder before real-time streaming was introduced.

This is a re-export of recorder.py as of the chunk-based pipeline.
Kept for rollback if the streaming pipeline needs to be reverted.
"""

# This file is a backup — import from recorder.py for active use.
from radios.analysis.recorder import *  # noqa: F401,F403
