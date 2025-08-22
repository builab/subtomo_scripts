#!/usr/bin/env python3
#run on .txt file (model2point -Object -Contour filename.mod filename.txt)
#python axoneme_length.py -i filename.txt -o filename_axolengths.txt --npix 7.77

import argparse
import sys
import math
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute curved axoneme length for each cilium."
    )
    p.add_argument("-i", "--input", required=True,
                   help="Input file (cols: micrograph, cilia, x, y, …)")
    p.add_argument("-o", "--output", required=True,
                   help="Output file: micrograph, cilia, axoneme_nm")
    p.add_argument("--npix", type=float, required=True,
                   help="Nanometres per pixel")
    p.add_argument("--threshold", type=float, default=20.0,
                   help="Pixel-distance threshold to drop accidental tip duplicates")
    return p.parse_args()

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def main():
    args = parse_args()
    data = defaultdict(list)

    # read and group points
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
                sys.stderr.write(f"Skip {micro}, cilium {cilium}: <2 points\n")
                continue

            # detect true tip index
            tip_idx = n - 1
            base_tip_idx = n - 2
            if euclid(pts[tip_idx], pts[base_tip_idx]) < args.threshold:
                tip_idx -= 1
                base_tip_idx = tip_idx - 1
                while base_tip_idx >= 0 and euclid(pts[tip_idx], pts[base_tip_idx]) < args.threshold:
                    base_tip_idx -= 1
                if base_tip_idx < 0:
                    sys.stderr.write(
                        f"Skip {micro}, cilium {cilium}: no valid tip found\n"
                    )
                    continue

            # sum consecutive distances up to tip_idx
            axoneme_px = sum(
                euclid(pts[i], pts[i+1]) for i in range(tip_idx)
            )
            axoneme_nm = axoneme_px * args.npix

            out.write(f"{micro}\t{cilium}\t{axoneme_nm:.3f}\n")

if __name__ == "__main__":
    main()
