# Usage example
# python modify_mdoc_framename.py --subframe_pattern CHE12over_ --offset 1000 input.mdoc output.mdoc > mv_frames.sh
# The --subframe_pattern to make sure getting file file pattern correctly.
# Assuming pattern of 5 digits in McGill, the new frame files with be + offset value.
# Better to make a bak_mdoc and then create the output file in mdoc.
# After that, check the mv command and run it with sh ./mv_frames.sh
# HB 2024/12/28


import re
import argparse

def modify_subframe_path(file_path, output_path, subframe_pattern, offset):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    move_commands = []

    for line in lines:
        # Check for the line with the specified SubFramePath pattern and modify it
        match = re.search(rf'(SubFramePath = .+?{subframe_pattern})(\d{{5}})(_.+\.tif)', line)
        if match:
            prefix = match.group(1)
            number = int(match.group(2))  # Extract the number
            suffix = match.group(3)
            new_number = number + offset  # Add the offset to the number
            modified_line = f"{prefix}{new_number:05d}{suffix}\n"
            modified_lines.append(modified_line)

            # Extract old and new file names for move command
            old_file = f"{prefix[len('SubFramePath = '):]}{number:05d}{suffix}".replace('\\', '/')
            new_file = f"{prefix[len('SubFramePath = '):]}{new_number:05d}{suffix}".replace('\\', '/')
            old_file_name = old_file.split('/')[-1]
            new_file_name = new_file.split('/')[-1]
            move_commands.append(f"mv \"Frames/{old_file_name}\" \"Frames/{new_file_name}\"")
        else:
            modified_lines.append(line)

    with open(output_path, 'w') as file:
        file.writelines(modified_lines)

    # Print move commands
    if move_commands:
        print("\n# Commands to rename files:")
        for command in move_commands:
            print(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify the frame name in an mdoc file.")
    parser.add_argument("--subframe_pattern", type=str, required=True, help="The SubFramePath pattern to search for.")
    parser.add_argument("--offset", type=int, required=True, help="The number to add to the matched pattern.")
    parser.add_argument("input_file", type=str, help="Path to the input file.")
    parser.add_argument("output_file", type=str, help="Path to the output file.")

    args = parser.parse_args()

    modify_subframe_path(args.input_file, args.output_file, args.subframe_pattern, args.offset)

