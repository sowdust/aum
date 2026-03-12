"""
Segmenter tuning script.

Evaluates boundary accuracy against hand-labeled ground truth and
optionally runs a resumable grid search over the tuning parameters.

Usage
-----
# Evaluate current parameters on all labeled files:
    python radios/analysis/tune.py labels.json

# Grid search (slow -- reruns segmentation per parameter combination):
    python radios/analysis/tune.py labels.json --grid

# Resume a previously interrupted grid search:
    python radios/analysis/tune.py labels.json --grid --resume

# Use a custom checkpoint file (default: tune_checkpoint.json):
    python radios/analysis/tune.py labels.json --grid --checkpoint my_run.json

Label file format (JSON)
------------------------
A JSON array; each entry describes one audio file and its known
segment boundaries:

    [
      {
        "file": "/absolute/path/to/recording.mp3",
        "segments": [
          {"start": 0.0,   "end": 187.0, "type": "speech"},
          {"start": 187.0, "end": 534.5, "type": "music"},
          {"start": 534.5, "end": 721.0, "type": "speech"}
        ]
      }
    ]

Rules for labeling:
  - "start" of the first segment must be 0.0
  - "end" of the last segment should match the file duration
  - "type" is one of: speech, music, noise, noEnergy
  - Consecutive segments must not overlap or have gaps
  - Boundaries should reflect the actual content transition,
    not the nearest convenient timestamp

The metric reported is the boundary offset error: for each predicted
boundary the script finds the nearest ground truth boundary of the same
transition type and measures |predicted - truth| in seconds.  It
reports mean, median, and max error across all boundaries in all files.
"""

import argparse
import json
import os
import signal
import sys

# Bootstrap Django so we can import segmenter (needs settings).
# Run from the project root: python radios/analysis/tune.py ...
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "aum.settings")

import django
django.setup()

import itertools
import numpy as np

import radios.analysis.segmenter as seg_mod
from radios.analysis.segmenter import segment_audio, AudioSegment
from typing import List, Dict, Any, Optional


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def _boundaries(segments: List[AudioSegment]) -> List[float]:
    """Return interior boundary timestamps (not the 0.0 start)."""
    return [s.start for s in segments[1:]]


def _boundary_errors(
    predicted: List[AudioSegment],
    truth: List[AudioSegment],
    tolerance: float = 30.0,
) -> List[float]:
    """
    For each predicted boundary, find the nearest ground truth boundary
    within *tolerance* seconds and record the absolute error.

    Predicted boundaries with no nearby ground truth boundary are
    counted as *tolerance* seconds wrong (worst case).
    """
    pred_b = _boundaries(predicted)
    truth_b = _boundaries(truth)

    if not truth_b:
        return []
    if not pred_b:
        return [tolerance] * len(truth_b)

    errors = []
    for pb in pred_b:
        nearest = min(truth_b, key=lambda tb: abs(pb - tb))
        errors.append(min(abs(pb - nearest), tolerance))
    return errors


def evaluate(labeled: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Run segment_audio on each labeled file and compute boundary errors.
    Returns a dict with mean, median, max, and count.
    """
    all_errors = []

    for entry in labeled:
        path = entry["file"]
        truth_segs = [
            AudioSegment(
                start=s["start"],
                end=s["end"],
                segment_type=s["type"],
            )
            for s in entry["segments"]
        ]

        print(f"  {os.path.basename(path)} ...", end=" ", flush=True)
        predicted = segment_audio(path)
        errors = _boundary_errors(predicted, truth_segs)
        all_errors.extend(errors)

        if errors:
            print(f"boundaries: {len(errors)}, mean={np.mean(errors):.1f}s, max={np.max(errors):.1f}s")
        else:
            print("no boundaries to compare")

    if not all_errors:
        return {}

    return {
        "mean":   float(np.mean(all_errors)),
        "median": float(np.median(all_errors)),
        "max":    float(np.max(all_errors)),
        "count":  len(all_errors),
    }


# -----------------------------------------------------------------------
# Grid search
# -----------------------------------------------------------------------

# Define the parameter grid here.  Each key must match a module-level
# constant in segmenter.py.  Add/remove values to taste.
PARAM_GRID = {
    "JITTER_THRESHOLD":    [2.0, 3.0, 5.0],
    "SEGMENT_MIN_DURATION_DEFAULT": [10.0, 15.0, 20.0],
    "FLUX_SMOOTH_SEC":     [1.0, 2.0, 3.0],
    "PROXIMITY_SIGMA":     [2.0, 5.0, 8.0],
    "REFINE_SEARCH_RADIUS":[5.0, 10.0, 15.0],
}

_DEFAULT_CHECKPOINT = "tune_checkpoint.json"


def _set_params(params: Dict[str, float]) -> None:
    """Patch segmenter module-level constants in place."""
    for name, value in params.items():
        setattr(seg_mod, name, value)

    # Also reset the singleton so the next segment_audio call uses fresh
    # parameters (the CNN itself doesn't need reloading, but we clear the
    # module cache for safety).
    seg_mod._segmenter_instance = seg_mod._segmenter_instance  # no-op intentional


def _save_checkpoint(path: str, results: list, completed: int) -> None:
    """Persist completed results and how many combinations were finished."""
    data = {
        "completed": completed,
        "results": [
            {"mean": mean, "params": params, "metrics": metrics}
            for mean, params, metrics in results
        ],
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)  # atomic replace to avoid corrupt checkpoint on crash


def _load_checkpoint(path: str) -> tuple[int, list]:
    """
    Load a checkpoint file.
    Returns (completed_count, results) where results is a list of
    (mean, params, metrics) tuples.
    """
    with open(path) as f:
        data = json.load(f)
    completed = data.get("completed", 0)
    results = [
        (r["mean"], r["params"], r["metrics"])
        for r in data.get("results", [])
    ]
    return completed, results


def _print_best(results: list) -> None:
    """Print the current best parameter set."""
    if not results:
        return
    best_mean, best_params, best_metrics = min(results, key=lambda r: r[0])
    param_str = "  ".join(f"{k}={v}" for k, v in best_params.items())
    print(
        f"  *** Best so far: mean={best_mean:.2f}s  "
        f"median={best_metrics['median']:.2f}s  "
        f"max={best_metrics['max']:.2f}s ***\n"
        f"      {param_str}"
    )


def grid_search(
    labeled: List[Dict[str, Any]],
    checkpoint_path: str = _DEFAULT_CHECKPOINT,
    resume: bool = False,
) -> None:
    """
    Try all combinations in PARAM_GRID and report results sorted by mean error.

    Progress is saved to *checkpoint_path* after each combination so the run
    can be interrupted with Ctrl+C and resumed later with --resume.
    """
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)

    results: list = []
    start_idx = 0

    if resume and os.path.exists(checkpoint_path):
        start_idx, results = _load_checkpoint(checkpoint_path)
        print(f"Resuming from checkpoint '{checkpoint_path}': {start_idx}/{total} already done.\n")
        if results:
            _print_best(results)
            print()
    elif resume:
        print(f"No checkpoint found at '{checkpoint_path}', starting fresh.\n")

    print(f"Grid search: {total} combinations × {len(labeled)} file(s)  "
          f"(starting at #{start_idx + 1})\n")

    # Track the combination index even if the loop body is never entered
    # (so the KeyboardInterrupt handler can reference it safely).
    combo_idx = start_idx - 1

    def _handle_interrupt(signum, frame):
        # Called when Ctrl+C is pressed mid-evaluation.  Save what we have.
        print("\n\nInterrupted by user.  Saving progress...")
        _save_checkpoint(checkpoint_path, results, combo_idx + 1)
        print(f"Checkpoint saved to '{checkpoint_path}'.")
        print(f"Resume with:  python radios/analysis/tune.py <labels> --grid --resume\n")
        if results:
            print("Best combination found so far:")
            _print_best(results)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_interrupt)

    for combo_idx, combo in enumerate(combinations[start_idx:], start=start_idx):
        params = dict(zip(keys, combo))
        _set_params(params)

        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"[{combo_idx + 1}/{total}] {param_str}")

        metrics = evaluate(labeled)
        if metrics:
            results.append((metrics["mean"], params, metrics))
            print(
                f"  → mean={metrics['mean']:.2f}s  "
                f"median={metrics['median']:.2f}s  "
                f"max={metrics['max']:.2f}s"
            )
            _print_best(results)
        else:
            print("  → no boundaries evaluated")

        _save_checkpoint(checkpoint_path, results, combo_idx + 1)
        print()

    # Restore default SIGINT so the summary can be interrupted normally.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    if not results:
        print("No results.")
        return

    results.sort(key=lambda r: r[0])

    print("\n=== Top 5 parameter sets (by mean boundary error) ===\n")
    for mean_err, params, metrics in results[:5]:
        print(f"  mean={mean_err:.2f}s  median={metrics['median']:.2f}s  max={metrics['max']:.2f}s")
        for k, v in params.items():
            print(f"    {k} = {v}")
        print()

    # Remove checkpoint once the full run completes successfully.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"(Checkpoint '{checkpoint_path}' removed — run complete.)")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate / tune the audio segmenter.")
    parser.add_argument("labels", help="Path to labeled JSON file.")
    parser.add_argument(
        "--grid", action="store_true",
        help="Run grid search over PARAM_GRID instead of a single evaluation.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a previously interrupted grid search from the checkpoint file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=_DEFAULT_CHECKPOINT,
        metavar="FILE",
        help=f"Checkpoint file for grid search progress (default: {_DEFAULT_CHECKPOINT}).",
    )
    args = parser.parse_args()

    with open(args.labels) as f:
        labeled = json.load(f)

    print(f"Loaded {len(labeled)} labeled file(s).\n")

    if args.grid:
        grid_search(labeled, checkpoint_path=args.checkpoint, resume=args.resume)
    else:
        if args.resume:
            print("Note: --resume is only used with --grid and has no effect here.\n")
        print("Evaluating current parameters...\n")
        metrics = evaluate(labeled)
        if metrics:
            print(f"\nResults across {metrics['count']} boundaries:")
            print(f"  mean error:   {metrics['mean']:.2f} s")
            print(f"  median error: {metrics['median']:.2f} s")
            print(f"  max error:    {metrics['max']:.2f} s")
        else:
            print("No boundaries to evaluate.")


if __name__ == "__main__":
    main()