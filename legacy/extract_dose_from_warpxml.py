#!/usr/bin/env python3
# Extract dose from warpxml for pytom_match_pick
# Script originated from Juha Huiskonen, modified by Huy Bui

import sys
import re
import os

# Check if at least one input file is given
if len(sys.argv) < 2:
    print("Usage: python extract_dose_from_warpxml.py input_file1 [input_file2 ...]")
    sys.exit(1)

# Loop over each input file
for input_file in sys.argv[1:]:
    # Output name: <basename>_dose.txt
    base_name = os.path.basename(input_file)
    base_name_no_ext = os.path.splitext(base_name)[0]
    output_file = f"{base_name_no_ext}_dose.txt"

    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            content = infile.read()
            # Extract content between <Dose> and </Dose>
            matches = re.findall(r'<Dose>(.*?)</Dose>', content, re.DOTALL)
            for match in matches:
                outfile.write(match.strip() + '\n')
        print(f"Extracted dose content written to {output_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
