# Script to copy xf and tlt files from existing IMOD alignment to the warp_tiltseries/tiltstack
# Read the align.com file for excluded view and filter that out from the xf and tlt before copying
# Usage:
# copy_xf_tlt_files_to_warp.py align.com tlt file xf file
# Written by ChatGPT, edited by HB and Avrin Ghanaeian

import os
import shutil
import argparse

def parse_exclude_list(exclude_string):
    """Parse the ExcludeList string into a list of integers."""
    result = []
    parts = exclude_string.split(',')
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result

def get_exclude_list(file_path):
    """Read the file and return the ExcludeList as a list of integers."""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            if line.strip().startswith("ExcludeList"):
                exclude_numbers = line.strip().split(maxsplit=1)[1]
                return parse_exclude_list(exclude_numbers)

        return []

    except FileNotFoundError:
        print(f"Error: The file at {file_path} does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def filter_and_invert_tilt_file(input_path, output_path, exclude_indices):
    """Filter a tilt file to exclude specific 1-based lines and invert the sign of remaining values."""
    try:
        with open(input_path, 'r') as infile:
            lines = infile.readlines()

        filtered_lines = []
        excluded_lines = []
        for idx, line in enumerate(lines, start=1):
            if idx in exclude_indices:
                excluded_lines.append(idx)
            else:
                try:
                    value = float(line.strip())
                    inverted_value = -value
                    filtered_lines.append(f"{inverted_value}\n")
                except ValueError:
                    print(f"Warning: Non-numeric value on line {idx}: {line.strip()}")

        with open(output_path, 'w') as outfile:
            outfile.writelines(filtered_lines)

        print(f"Filtered and inverted tilt file written to {output_path}")
        print(f"Excluded lines: {excluded_lines}")

    except FileNotFoundError:
        print(f"Error: The file at {input_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def filter_text_file(input_path, output_path, exclude_indices):
    """Filter a text file (e.g., .xf file) to exclude specific 1-based lines."""
    try:
        with open(input_path, 'r') as infile:
            lines = infile.readlines()

        if not lines:
            print(f"Warning: The input file {input_path} is empty.")
            return

        filtered_lines = []
        excluded_lines = []
        for idx, line in enumerate(lines, start=1):
            if idx in exclude_indices:
                excluded_lines.append(idx)
            else:
                # Ensure the line content is retained correctly
                filtered_lines.append(line.strip() + "\n")

        if not filtered_lines:
            print(f"Warning: No valid lines remain in {input_path} after filtering.")

        with open(output_path, 'w') as outfile:
            outfile.writelines(filtered_lines)

        print(f"Filtered text file written to {output_path}")
        print(f"Excluded lines: {excluded_lines}")

    except FileNotFoundError:
        print(f"Error: The file at {input_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Filter and copy .tlt and .xf files.")
    parser.add_argument("align_com_file", help="Path to the align.com file.")
    parser.add_argument("input_tilt_file", help="Path to the input tilt file.")
    parser.add_argument("input_xf_file", help="Path to the input xf file.")
    args = parser.parse_args()

    align_com_file = args.align_com_file
    input_tilt_file = args.input_tilt_file
    input_xf_file = args.input_xf_file

    # Output directory
    output_directory = "warp_tiltseries/tiltstack"
    os.makedirs(output_directory, exist_ok=True)

    # Define output paths
    output_tilt_file = os.path.join(output_directory, os.path.basename(input_tilt_file))
    output_xf_file = os.path.join(output_directory, os.path.basename(input_xf_file))

    exclude_list = get_exclude_list(align_com_file)
    filter_and_invert_tilt_file(input_tilt_file, output_tilt_file, exclude_list)
    filter_text_file(input_xf_file, output_xf_file, exclude_list)

if __name__ == "__main__":
    main()

