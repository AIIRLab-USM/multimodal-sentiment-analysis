#!/usr/bin/env python3
"""
WikiArt Duplicate Filename Scanner (one-time version)
-----------------------------------------------------
Traverses a local WikiArt directory (./wikiart), finds duplicate filenames
across genres, and produces useful statistics + CSVs.

Outputs:
  - duplicate_report/duplicates.csv
  - duplicate_report/genre_summary.csv
  - duplicate_report/genre_pair_summary.csv
  - duplicate_report/overall.txt
"""

import os
import csv
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path("./wikiart")
OUTDIR = Path("./data")

def get_genre(root: Path, file_path: Path) -> str:
    """Assumes first-level directory under root is the 'genre'."""
    rel = file_path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "(root)"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # key -> list of (path, genre, name_original)
    index = defaultdict(list)
    total_files_by_genre = Counter()

    # Walk the tree
    for dirpath, _, filenames in os.walk(ROOT):
        for fname in filenames:
            p = Path(dirpath) / fname
            if not is_image(p):
                continue
            genre = get_genre(ROOT, p)
            total_files_by_genre[genre] += 1

            key = p.stem.lower()
            index[key].append((p, genre, fname))

    # Duplicate groups
    duplicate_groups = {k: v for k, v in index.items() if len(v) > 1}

    # Per-genre duplicate counts
    duplicate_files_by_genre = Counter()
    for entries in duplicate_groups.values():
        for _, genre, _ in entries:
            duplicate_files_by_genre[genre] += 1

    # duplicates.csv
    with (OUTDIR / "duplicates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["basename", "stem_lower", "n_copies", "genres", "paths"])
        for key, entries in sorted(duplicate_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            genres = sorted({g for _, g, _ in entries})
            paths = [str(p) for p, _, _ in entries]
            basename = entries[0][2]
            writer.writerow([basename, key, len(entries), ";".join(genres), "|".join(paths)])


if __name__ == "__main__":
    main()
