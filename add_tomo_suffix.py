#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_tomo_prefix.py - Rename tomography .mdoc files (or any files).
Add prefix to mdoc files & mrc file to make them unique different days.
In addition, it also change the name of the Tomo5 Position_1.mrc to Position_1_1.mrc for easier processing later
Usage: add_tomo_prefix.py --prefix 20260923_ --apply_tomo5_pattern Position*.mdoc

Options:
  --prefix PREFIX         Prepend PREFIX to each filename
                          (Position_1.mdoc -> 20260923_Position_1.mdoc)
  --apply_tomo5_pattern   Make names uniform with the Tomo5 pattern
                          Position_2_2.mdoc -> unchanged
                          Position_2.mdoc   -> Position_2_1.mdoc
  Both can be combined.

Examples:
  add_tomo_prefix.py --prefix 20260923_ Position*.mdoc
  add_tomo_prefix.py --apply_tomo5_pattern Position*.mdoc
  add_tomo_prefix.py --prefix 20260923_ --apply_tomo5_pattern Position*.mdoc
  add_tomo_prefix.py --prefix 20260923_ --dry_run Position*.mdoc
"""

import argparse
import re
import sys
from pathlib import Path

# Matches names like Position_2.mdoc / Position_2_2.mdoc / Position_2.mrc.mdoc
# Group 1: everything up to and including the position number
# Group 2: optional "_N" suffix (the Tomo5 tilt-series/sub-index)
# Group 3: remainder (extension(s))
TOMO5_RE = re.compile(r"^(.*?\d+)(_\d+)?((?:\.[^.]+)+)$")


def apply_tomo5(name: str) -> str:
    """Add '_1' to names lacking the Tomo5 '_N' suffix; leave others untouched."""
    m = TOMO5_RE.match(name)
    if not m:
        return name
    base, suffix, ext = m.groups()
    if suffix:  # already has the _N pattern
        return name
    return f"{base}_1{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Add a prefix and/or apply the Tomo5 naming pattern to files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--prefix", default="",
                        help="Prefix to prepend to each filename (e.g. 20260923_)")
    parser.add_argument("--apply_tomo5_pattern", action="store_true",
                        help="Rename Position_N.mdoc to Position_N_1.mdoc "
                             "(files already like Position_N_M.mdoc are untouched)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be renamed without doing it")
    parser.add_argument("files", nargs="+", help="Files to rename (e.g. Position*.mdoc)")
    args = parser.parse_args()

    if not args.prefix and not args.apply_tomo5_pattern:
        parser.error("Nothing to do: give --prefix and/or --apply_tomo5_pattern")

    # Plan all renames first
    plan = []
    for f in args.files:
        src = Path(f)
        if not src.is_file():
            print(f"Skipping (not a file): {f}", file=sys.stderr)
            continue

        name = src.name
        if args.apply_tomo5_pattern:
            name = apply_tomo5(name)
        if args.prefix and not name.startswith(args.prefix):
            name = args.prefix + name

        dst = src.with_name(name)
        if dst == src:
            print(f"Unchanged: {src.name}")
            continue
        plan.append((src, dst))

    # Safety checks: collisions with existing files or between targets
    targets = [d for _, d in plan]
    sources = {s for s, _ in plan}
    errors = []
    if len(set(targets)) != len(targets):
        errors.append("Multiple files would be renamed to the same name.")
    for dst in targets:
        if dst.exists() and dst not in sources:
            errors.append(f"Target already exists: {dst.name}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print("No files were renamed.", file=sys.stderr)
        sys.exit(1)

    # Two-step rename via temp names to handle chained renames safely
    # (e.g. Position_2.mdoc -> Position_2_1.mdoc when Position_2_1.mdoc is also being renamed)
    if args.dry_run:
        for src, dst in plan:
            print(f"[dry run] {src.name} -> {dst.name}")
        return

    temps = []
    for i, (src, dst) in enumerate(plan):
        tmp = src.with_name(f".{src.name}.tmp_rename_{i}")
        src.rename(tmp)
        temps.append((tmp, src, dst))
    for tmp, src, dst in temps:
        tmp.rename(dst)
        print(f"{src.name} -> {dst.name}")

    print(f"Renamed {len(plan)} file(s).")


if __name__ == "__main__":
    main()