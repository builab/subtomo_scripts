#!/usr/bin/env python3

import sys
import csv
import imodmodel
import os

def check_sorting_order(points):
    """Check if Y values are sorted ascending or descending."""
    y_values = [p[1] for p in points]

    if y_values == sorted(y_values):
        return 0  # ascending
    elif y_values == sorted(y_values, reverse=True):
        return 1  # descending
    else:
        return -1  # unsorted or equal

def process_mod_file(filepath):
    """Process one .mod file and return rows."""
    model = imodmodel.read(filepath)
    base_name = os.path.basename(filepath)
    rows = []

    for obj in model.objects:
        for contour_num, contour in enumerate(obj.contours):
            sorting_order = check_sorting_order(contour.points)
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

