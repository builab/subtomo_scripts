#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# Need to fix the image view

import os
import glob
from chimerax.core.commands import run

# Define the folder path and the pattern
folder_path = ''  # Replace with the actual path to your folder containing all cmm files
tomo_path = ''  # Replace with the actual path to your folder containing all mrc tomograms
pattern = 'CHE12over*'

# Get all files in the folder that match the pattern
files = glob.glob(os.path.join(folder_path, pattern))

# Group files by the unique part of the pattern
groups = {}
for file in files:
    # Extract the base name of the file (without the folder path)
    base_name = os.path.basename(file)
    
    # Extract the group key (e.g., CHE12over_01_14.00Apx)
    group_key = '_'.join(base_name.split('_')[:3])
    
    # Add the file to the corresponding group
    if group_key not in groups:
        groups[group_key] = []
    groups[group_key].append(base_name)

# Process each group
for group_key, file_list in groups.items():
    print(f"Group: {group_key}")
    
    # Combine all input files for this group
    input_files = " ".join([os.path.join(folder_path, file_name) for file_name in file_list])
    print(f"Opening: {input_files}")
    
    # Create output filenames
    output_png = group_key + '.png'
    output_tomo_png = group_key + '_tomo.png'
    
    # Run in ChimeraX - First save the cmm model image
    run(session, 'close all')
    run(session, f'open {input_files}')
    # run(session, f'save {output_png} width 1000 super 3') # Uncomment this line to save model-only previews
    
    # Try to open the tomogram file
    try:
        # Correct format with the plus sign: group_key+group_key_refined.mrc
        tomo_file = os.path.join(tomo_path, f"{group_key}+{group_key}_refined.mrc")
        print(f"Attempting to open tomogram: {tomo_file}")
        
        # Check if the file exists before trying to open it
        if os.path.exists(tomo_file):
            # Use quotes around the filename to handle special characters like the plus sign
            run(session, f'open "{tomo_file}"')
            run(session, 'view')
            run(session, 'volume #2 style image transparency .5')   
            run(session, f'save {output_tomo_png} width 1000 super 3')
        else:
            print(f"Warning: Tomogram file not found at {tomo_file}")
            
            # Try alternative filename patterns if the first one doesn't exist
            alt_patterns = [
                f"{group_key}.mrc",
                f"{group_key}_refined.mrc"
            ]
            
            found = False
            for pattern in alt_patterns:
                alt_tomo_file = os.path.join(tomo_path, pattern)
                if os.path.exists(alt_tomo_file):
                    print(f"Found alternative tomogram file: {alt_tomo_file}")
                    run(session, f'open "{alt_tomo_file}"')
                    run(session, 'view')
                    run(session, 'volume #2 style image transparency .5')   
                    run(session, f'save {output_tomo_png} width 1000 super 3')
                    found = True
                    break
                    
            if not found:
                print(f"Warning: No matching tomogram file found for {group_key}")
    
    except Exception as e:
        print(f"Error processing tomogram for {group_key}: {str(e)}")
        # Continue with the next group rather than stopping the entire script
        continue

print("Processing complete!")