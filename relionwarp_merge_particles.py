#!/usr/bin/env python3
import argparse
import pandas as pd
import starfile

# Script to merge particle star files from two datasets (no overlapped tomo name)

def load_star(path):
    """Read a RELION .star file with 3 blocks:
       data_general, data_optics, data_particles
       (starfile strips 'data_' so keys are 'general', 'optics', 'particles')
    """
    d = starfile.read(path)
    try:
        general = d["general"]
        optics = d["optics"]
        particles = d["particles"]
    except KeyError as e:
        raise RuntimeError(f"Missing expected block {e} in {path}")
    return general, optics, particles

def merge_stars(fileA, fileB, outfile):
    genA, optA, partA = load_star(fileA)
    genB, optB, partB = load_star(fileB)

    # --- merge optics ---
    optics_all = pd.concat([optA, optB], ignore_index=True)
    optics_all = optics_all.reset_index(drop=True)

    # Renumber optics groups sequentially
    optics_all["rlnOpticsGroup"] = range(1, len(optics_all) + 1)
    optics_all["rlnOpticsGroupName"] = [
        f"opticsGroup{i}" for i in optics_all["rlnOpticsGroup"]
    ]

    # --- build mapping from old → new ---
    nA = len(optA)
    mapA = dict(zip(optA["rlnOpticsGroup"], optics_all.loc[: nA - 1, "rlnOpticsGroup"]))
    mapB = dict(zip(optB["rlnOpticsGroup"], optics_all.loc[nA:, "rlnOpticsGroup"]))

    # --- update particles ---
    partA_new = partA.copy()
    partA_new["rlnOpticsGroup"] = partA_new["rlnOpticsGroup"].map(mapA)

    partB_new = partB.copy()
    partB_new["rlnOpticsGroup"] = partB_new["rlnOpticsGroup"].map(mapB)

    particles_all = pd.concat([partA_new, partB_new], ignore_index=True)

    # --- output STAR ---
    out_dict = {
        "data_general": genA,       # keep general from fileA
        "data_optics": optics_all,
        "data_particles": particles_all,
    }
    starfile.write(out_dict, outfile, overwrite=True)

    print(f"Merged {fileA} + {fileB} → {outfile}")
    print(f"Total optics groups: {len(optics_all)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge two RELION particle STAR files")
    ap.add_argument("--input", "-i", required=True, help="First STAR file")
    ap.add_argument("--input2", "-i2", required=True, help="Second STAR file")
    ap.add_argument("--output", "-o", required=True, help="Output STAR file")
    args = ap.parse_args()

    merge_stars(args.input, args.input2, args.output)
