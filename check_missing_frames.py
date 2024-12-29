# Script to check if frames are all present.
# Usage:
# python check_missing_frames.py --frame_dir Frames *.mdoc
# 2024/12/28

import os
import re
import argparse

def check_missing_frames(file_paths, frame_dir):
    missing_files = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # Check for the line with the SubFramePath pattern
            match = re.search(r'SubFramePath = .+\\(.+\.tif)', line)
            if match:
                # Extract the frame file name
                frame_name = match.group(1)
                frame_path = os.path.join(frame_dir, frame_name)

                # Check if the frame file exists
                if not os.path.exists(frame_path):
                    missing_files.append((frame_name, file_path))

    # Print missing files
    if missing_files:
        print("\n# Missing files:")
        for frame_name, mdoc in missing_files:
            print(f"Missing frame file: {frame_name} in mdoc: {mdoc}")
    else:
        print("All frame files are present.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for missing frame files in the specified directory.")
    parser.add_argument("input_files", type=str, nargs='+', help="Paths to the input mdoc files.")
    parser.add_argument("--frame_dir", type=str, default="Frames", help="Directory to check for frame files (default: Frames).")

    args = parser.parse_args()

    check_missing_frames(args.input_files, args.frame_dir)

