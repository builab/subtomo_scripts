import os
import shutil
import re
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

def backup_input_star_file(input_star_file):
    """Copy the input_star_file to a backup directory named bak_tomostar."""
    backup_dir = "bak_tomostar"
    os.makedirs(backup_dir, exist_ok=True)

    src = input_star_file
    dst = os.path.join(backup_dir, os.path.basename(input_star_file))
    shutil.copy2(src, dst)
    print(f"Copied {src} to {dst}")

def filter_star_file(input_path, exclude_indices):
    """Filter the STAR file to exclude data lines for specific 1-based indices."""
    try:
        with open(input_path, 'r') as infile:
            lines = infile.readlines()

        header = []
        data_lines = []
        is_data_section = False

        for line in lines:
            if line.strip() == "loop_":
                is_data_section = True
                header.append(line)
            elif not is_data_section:
                header.append(line)
            elif is_data_section and line.strip().startswith("_"):
                header.append(line)
            elif is_data_section and line.strip().startswith("../"):
                data_lines.append(line)

        filtered_lines = []
        excluded_lines = []
        for idx, line in enumerate(data_lines, start=1):
            if idx in exclude_indices:
                excluded_lines.append(idx)
            else:
                filtered_lines.append(line)

        # Overwrite the input file with the filtered content
        with open(input_path, 'w') as outfile:
            outfile.writelines(header)
            outfile.writelines(filtered_lines)

        print(f"Filtered STAR file written to {input_path}")
        print(f"Excluded indices: {excluded_lines}")

    except FileNotFoundError:
        print(f"Error: The file at {input_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process and filter a tomostar file.")
    parser.add_argument("input_star_file", help="Path to the input STAR file.")
    parser.add_argument("align_com_file", help="Path to the align.com file.")
    args = parser.parse_args()

    input_star_file = args.input_star_file
    align_com_file = args.align_com_file

    # Backup and process the file
    backup_input_star_file(input_star_file)
    exclude_list = get_exclude_list(align_com_file)
    filter_star_file(input_star_file, exclude_list)

if __name__ == "__main__":
    main()

