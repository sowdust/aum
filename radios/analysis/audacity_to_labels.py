"""
Convert an Audacity label export to the JSON format expected by tune.py.

Usage
-----
    python radios/analysis/audacity_to_labels.py \\
        labels.txt /absolute/path/to/recording.mp3 [output.json]

Arguments
---------
labels.txt          Audacity label export (tab-separated: start end name).
recording.mp3       The audio file the labels belong to.
output.json         Optional output path (default: labels.json).
                    If the file already exists the new entry is APPENDED
                    to the array, so you can label multiple files one by one.

Audacity label file format
--------------------------
Audacity exports region labels as a plain-text tab-separated file:

    0.000000    187.000000    speech
    187.000000  534.500000    music
    534.500000  721.000000    speech

Each row = one segment: start(s) TAB end(s) TAB type.
Type must be one of: speech, music, noise, noEnergy.

Generating the file in Audacity
--------------------------------
1. Open your recording in Audacity.
2. Add a label track: Tracks > Add New > Label Track.
3. For each segment:
   a. Click and drag in the waveform to select the region.
   b. Press Ctrl+B (Cmd+B on Mac) — a region label appears.
   c. Type the content type: speech, music, noise, or noEnergy.
4. When done, export: File > Export > Export Labels.
   Save the file anywhere (e.g. recording.txt).
5. Run this script to convert it.
"""

import json
import os
import sys
from typing import List, Dict, Any

VALID_TYPES = {"speech", "music", "noise", "noEnergy"}


def parse_audacity_labels(path: str) -> List[Dict[str, Any]]:
    segments = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                print(f"  Warning: line {lineno} has fewer than 3 tab-separated fields, skipped: {line!r}")
                continue

            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                print(f"  Warning: line {lineno} has non-numeric timestamps, skipped: {line!r}")
                continue

            label = parts[2].strip()
            if label not in VALID_TYPES:
                print(f"  Warning: line {lineno} unknown type {label!r} (expected one of {sorted(VALID_TYPES)})")

            if start >= end:
                print(f"  Warning: line {lineno} start >= end ({start} >= {end}), skipped.")
                continue

            segments.append({"start": start, "end": end, "type": label})

    return segments


def main():
    if len(sys.argv) < 3:
        print("Usage: python audacity_to_labels.py labels.txt /path/to/audio.mp3 [output.json]")
        sys.exit(1)

    labels_path = sys.argv[1]
    audio_path = os.path.abspath(sys.argv[2])
    output_path = sys.argv[3] if len(sys.argv) > 3 else "labels.json"

    print(f"Parsing {labels_path} ...")
    segments = parse_audacity_labels(labels_path)

    if not segments:
        print("No valid segments found. Check the label file format.")
        sys.exit(1)

    # Basic sanity checks
    for i, seg in enumerate(segments[:-1]):
        gap = segments[i + 1]["start"] - seg["end"]
        if abs(gap) > 0.1:
            print(f"  Warning: gap of {gap:.2f}s between segment {i} and {i+1}")

    print(f"Found {len(segments)} segments.")

    entry = {"file": audio_path, "segments": segments}

    # Load existing file if present (append mode)
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        # Replace entry for same file if it already exists
        existing = [e for e in existing if e["file"] != audio_path]
        existing.append(entry)
        data = existing
        print(f"Updated entry for {audio_path} in {output_path}.")
    else:
        data = [entry]
        print(f"Created {output_path}.")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone. {output_path} now contains {len(data)} file(s).")
    print(f"Next step: python radios/analysis/tune.py {output_path}")


if __name__ == "__main__":
    main()
