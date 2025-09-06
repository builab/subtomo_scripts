#!/usr/bin/env python3
import argparse
import pandas as pd
import starfile

def load_star(path):
    """Read any RELION .star file into a dict of DataFrames."""
    return starfile.read(path)

def merge_stars(fileA, fileB, outfile):
    dA = load_star(fileA)
    dB = load_star(fileB)

    out_dict = {}

    # --- merge global ---
    if "global" not in dA or "global" not in dB:
        raise RuntimeError("Both files must contain data_global block")

    globA = dA["global"]
    globB = dB["global"]

    global_all = pd.concat([globA, globB], ignore_index=True).reset_index(drop=True)

    # Reassign new sequential opticsGroupName
    new_names = [f"opticsGroup{i}" for i in range(1, len(global_all) + 1)]
    global_all["rlnOpticsGroupName"] = new_names

    out_dict["global"] = global_all

    # --- copy all other blocks (no overlap, just append if same name) ---
    for key, val in dA.items():
        if key != "global":
            out_dict[f"{key}"] = val
    for key, val in dB.items():
        if key != "global":
            # If block name already exists, stack them
            if f"{key}" in out_dict:
                out_dict[f"{key}"] = pd.concat(
                    [out_dict[f"{key}"], val], ignore_index=True
                )
            else:
                out_dict[f"{key}"] = val

    # --- write output ---
    starfile.write(out_dict, outfile, overwrite=True)

    print(f"Merged {fileA} + {fileB} → {outfile}")
    print(f"Total global entries: {len(global_all)}")
    print(f"Blocks written: {list(out_dict.keys())}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge two RELION STARs with data_global remapped")
    ap.add_argument("--input", "-i", required=True, help="First STAR file")
    ap.add_argument("--input2", "-i2", required=True, help="Second STAR file")
    ap.add_argument("--output", "-o", required=True, help="Output STAR file")
    args = ap.parse_args()

    merge_stars(args.input, args.input2, args.output)
