#!/usr/bin/env python3
# Extract defocus from warpxml for pytom_match_pick
# Script originated from Juha Huiskonen, modified by Huy Bui

import sys
import re
import os

# Check if at least one input file is given
if len(sys.argv) < 2:
    print("Usage: python extract_defocus_from_warpxml.py input_file1 [input_file2 ...]")
    sys.exit(1)

# Loop over each input file given
for input_file in sys.argv[1:]:
    # Generate output filename: <basename>_defocus.txt
    base_name = os.path.basename(input_file)
    base_name_no_ext = os.path.splitext(base_name)[0]
    output_file = f"{base_name_no_ext}_defocus.txt"

    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            content = infile.read()
            # Match only <GridCTF> blocks and extract <Node> values
            grid_blocks = re.findall(r'<GridCTF\b.*?>(.*?)</GridCTF>', content, re.DOTALL)
            for block in grid_blocks:
                node_matches = re.findall(r'<Node\b.*?Value="([^"]+)".*?/>', block)
                for value in node_matches:
                    outfile.write(value + '\n')
        print(f"Extracted Node values written to {output_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

