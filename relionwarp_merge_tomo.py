#!/usr/bin/env python3
import argparse
import pandas as pd
import starfile

# Script to merge tomo star files from star files from two datasets (no overlapped tomo name)

def load_star(path):
    """Read a RELION .star file with only data_global block"""
    d = starfile.read(path)
    try:
        global_block = d["global"]
    except KeyError:
        raise RuntimeError(f"Missing data_global block in {path}")
    return global_block

def merge_stars(fileA, fileB, outfile):
    globA = load_star(fileA)
    globB = load_star(fileB)

    # --- merge global ---
    global_all = pd.concat([globA, globB], ignore_index=True)
    global_all = global_all.reset_index(drop=True)

    # Reassign new sequential opticsGroupName
    new_names = [f"opticsGroup{i}" for i in range(1, len(global_all) + 1)]
    global_all["rlnOpticsGroupName"] = new_names

    # --- output STAR ---
    out_dict = {
        "global": global_all
    }
    starfile.write(out_dict, outfile, overwrite=True)

    print(f"Merged {fileA} + {fileB} → {outfile}")
    print(f"Total optics groups: {len(global_all)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge two RELION global STAR files")
    ap.add_argument("--input", "-i", required=True, help="First STAR file")
    ap.add_argument("--input2", "-i2", required=True, help="Second STAR file")
    ap.add_argument("--output", "-o", required=True, help="Output STAR file")
    args = ap.parse_args()

    merge_stars(args.input, args.input2, args.output)
