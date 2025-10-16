# filament_model_stats.py
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict
import tempfile
import os

"""
Quantify filament length from IMOD files
Usage: filament_model_stats.py --angpix 8.48 *.mod

@Builab 2025
@Author: Molly Yu
"""



def read_model2point(mod_path):
    fd, tmp_path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)  # Close the file descriptor so model2point can write to it

    try:
        subprocess.run([
            "model2point",
            "-input", str(mod_path),
            "-output", tmp_path,
            "-float", "-object", "-contour", "-zero"
        ], check=True)

        points = []
        with open(tmp_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) >= 5:
                    obj, cont = int(tokens[0]), int(tokens[1])
                    x, y, z = map(float, tokens[2:5])
                    points.append((obj, cont, x, y, z))
    finally:
        os.remove(tmp_path)

    return points

def compute_lengths(points, angpix):
    filaments = defaultdict(list)
    for obj, cont, x, y, z in points:
        filaments[(obj, cont)].append((x, y, z))

    lengths = []
    for (obj, cont), pts in filaments.items():
        pts = np.array(pts)
        if len(pts) < 2:
            continue  # not enough points to measure length
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        length_ang = dists.sum() * angpix
        lengths.append({
            "object": obj,
            "contour": cont,
            "length": length_ang
        })
    return lengths

def main():
    parser = argparse.ArgumentParser(description="Analyze IMOD filament models")
    parser.add_argument("--angpix", type=float, required=True, help="Angstroms per pixel")
    parser.add_argument("mod_files", nargs="+", help="List of .mod files")
    args = parser.parse_args()

    all_lengths = []
    filament_counts = defaultdict(int)

    output_dir = Path("filament_stats_output")
    output_dir.mkdir(exist_ok=True)

    for mod_file in args.mod_files:
        mod_path = Path(mod_file)
        print(f"Processing {mod_path.name}...")
        points = read_model2point(mod_path)
        lengths = compute_lengths(points, args.angpix)

        for entry in lengths:
            entry["tomogram"] = mod_path.stem
            all_lengths.append(entry)
            filament_counts[mod_path.stem] += 1

    df = pd.DataFrame(all_lengths)

    if df.empty:
        print("No filament data was found in the input files.")
        return

    # Histogram of filament lengths per cluster (object)
    plt.figure()
    for obj_id, group in df.groupby("object"):
        plt.hist(group["length"], bins=20, alpha=0.6, label=f"Cluster {obj_id}")
    plt.xlabel("Filament length (Å)")
    plt.ylabel("Count")
    plt.title("Filament Length Distribution per Cluster")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hist_filament_lengths.png")

    # Bar chart: number of filaments per tomogram
    plt.figure()
    tomo_names = list(filament_counts.keys())
    counts = list(filament_counts.values())
    plt.bar(tomo_names, counts)
    plt.xlabel("Tomogram")
    plt.ylabel("Number of filaments")
    plt.title("Filament Count per Tomogram")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "bar_filaments_per_tomogram.png")

    # Optional CSV export
    df.to_csv(output_dir / "filament_lengths.csv", index=False)
    pd.DataFrame({"tomogram": tomo_names, "count": counts}).to_csv(output_dir / "filament_counts.csv", index=False)

    print(f"Done. Outputs saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
