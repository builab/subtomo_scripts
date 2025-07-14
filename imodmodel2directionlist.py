#!/usr/bin/env python3

import sys
import csv
import imodmodel
import os

def check_sorting_order(y_values):
    """Check if Series of Y is ascending or descending."""
    if y_values.equals(y_values.sort_values()):
        return 0  # ascending
    elif y_values.equals(y_values.sort_values(ascending=False)):
        return 1  # descending
    else:
        return -1  # unsorted

def process_mod_file(filepath):
    """Process one .mod file and return rows."""
    df = imodmodel.read(filepath)
    base_name = os.path.basename(filepath)
    rows = []

    # Filter object_id == 0
    df_obj0 = df[df["object_id"] == 0]
    
    for contour_id in df_obj0["contour_id"].unique():
        contour_points = df_obj0[df_obj0["contour_id"] == contour_id]
        y_values = contour_points["y"].reset_index(drop=True)
        sorting_order = check_sorting_order(y_values)
        if sorting_order == -1:
            print(f"WARNING: {base_name} has sorting order of -1")
        rows.append([base_name, contour_num, sorting_order])

    return rows

def main():
    if len(sys.argv) < 2:
        print("Usage: python imodmodel2direction.py *.mod")
        sys.exit(1)

    mod_files = sys.argv[1:]

    all_rows = [["base_filename", "contour_number", "sorting_order"]]

    for mod_file in sorted(mod_files):
        rows = process_mod_file(mod_file)
        all_rows.extend(rows)

    with open("ciliaDirection.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    print("Wrote ciliaDirection.csv")

if __name__ == "__main__":
    main()

