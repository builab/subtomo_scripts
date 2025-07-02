#!/usr/bin/env python3
# Usage: mdoc2rawtlt.py A.mrc.mdoc
# HB, 2025/07/02

import sys
import os

def parse_mdoc(mdoc_path):
    angles = []
    with open(mdoc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('TiltAngle ='):
                _, val = line.split('=')
                angles.append(float(val.strip()))
    return angles

def write_rawtlt(output_path, angles):
    with open(output_path, 'w') as f:
        for angle in angles:
            f.write(f"{angle}\n")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <mdocfile>")
        sys.exit(1)
    
    mdoc_path = sys.argv[1]
    if not os.path.isfile(mdoc_path):
        print(f"Error: File '{mdoc_path}' does not exist.")
        sys.exit(1)
    
    # Strip .mrc.mdoc correctly
    if mdoc_path.endswith('.mrc.mdoc'):
        base = mdoc_path[:-len('.mrc.mdoc')]
    else:
        base = os.path.splitext(mdoc_path)[0]

    output_path = f"{base}.rawtlt"

    angles = parse_mdoc(mdoc_path)
    write_rawtlt(output_path, angles)
    print(f"Wrote {len(angles)} tilt angles to {output_path}")

if __name__ == '__main__':
    main()
