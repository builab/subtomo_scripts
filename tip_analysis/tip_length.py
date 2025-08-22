#!/usr/bin/env python3
#run on .txt file (model2point -Object -Contour filename.mod filename.txt)
#python tip_length.py -i filename.txt -o filename_tiplengths.txt --npix 7.77

import argparse
import sys
import math
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute tip length, dropping any accidental clicks closer than threshold."
    )
    p.add_argument("-i", "--input", required=True,
                   help="Input text file (cols: micrograph, cilia, x, y, …)")
    p.add_argument("-o", "--output", required=True,
                   help="Output text file with cols: micrograph, cilia, length_nm")
    p.add_argument("--npix", type=float, required=True,
                   help="Nanometres per pixel")
    p.add_argument("--threshold", type=float, default=20.0,
                   help="Pixel‐distance threshold for accidental clicks (default: 20.0)")
    return p.parse_args()

def euclid(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def main():
    args = parse_args()

    # group points by (micrograph, cilium)
    data = defaultdict(list)
    with open(args.input) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            micro, cilium = parts[0], parts[1]
            x, y = float(parts[2]), float(parts[3])
            data[(micro, cilium)].append((x, y))

    with open(args.output, "w") as out:
        for (micro, cilium), pts in sorted(data.items(),
                                          key=lambda t: (int(t[0][0]), int(t[0][1]))):
            n = len(pts)
            if n < 2:
                sys.stderr.write(f"Skip {micro}, cilium {cilium}: fewer than 2 points\n")
                continue

            # initial check: last vs second‐last
            d_last = euclid(pts[n-1], pts[n-2])
            if d_last >= args.threshold:
                tip_idx, base_idx = n-1, n-2
            else:
                # drop the duplicate at the tip, now tip is second‐last
                tip_idx = n-2
                base_idx = n-3
                # step back until distance ≥ threshold
                while base_idx >= 0 and euclid(pts[tip_idx], pts[base_idx]) < args.threshold:
                    base_idx -= 1

                if base_idx < 0:
                    sys.stderr.write(
                        f"Skip {micro}, cilium {cilium}: no two points ≥ {args.threshold}px apart\n"
                    )
                    continue

            dist_px = euclid(pts[tip_idx], pts[base_idx])
            length_nm = dist_px * args.npix
            out.write(f"{micro}\t{cilium}\t{length_nm:.3f}\n")

if __name__ == "__main__":
    main()
