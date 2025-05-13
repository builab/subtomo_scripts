# Script to modify ExposureDose inline (0 by default in McGill) to use Dose Filtering in IMOD
# Usage:
# python modify_mdoc_expdose.py --exposuredose 4.2 TS_01.mrc.mdoc TS_02.mrc.mdoc
# Badly written by ChatGPT, frustratedly fixed by HB, 2024/12

import argparse
import re
import sys

def modify_exposure_dose(file_path, exposure_dose):
    try:
        # Read the file content
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Modify lines containing 'ExposureDose'
        repl_str = r'\1 ' + str(exposure_dose)
        updated_lines = [
            re.sub(r'(ExposureDose\s*=)\s+\d+(\.\d+)?', repl_str, line)
            if 'ExposureDose' in line else line
            for line in lines
        ]

        # Write back the modified content to the same file
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)

        print(f"Successfully updated 'ExposureDose' to {exposure_dose} in {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify 'ExposureDose' values in .mdoc files.")
    parser.add_argument('--exposuredose', type=float, required=True, help="The new value for 'ExposureDose'.")
    parser.add_argument('files', nargs='+', type=str, help="Paths to the .mdoc files.")

    args = parser.parse_args()

    for file_path in args.files:
        modify_exposure_dose(file_path, args.exposuredose)
