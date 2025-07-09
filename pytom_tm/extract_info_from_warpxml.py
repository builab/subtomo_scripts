#!/usr/bin/env python3
# Extract dose, defocus and tilt angle from Warp XML for pytom_match_pick
# For McGill dataset, we flip the tilt angle to be compatible with IMOD
# Script by Huy Bui

import xml.etree.ElementTree as ET
import sys
import os

def write_values_to_file(output_file, values, label="Values"):
    """Write a list of values to a file, one per line."""
    with open(output_file, 'w') as f:
        for value in values:
            f.write(str(value) + '\n')
    print(f"{label} written to {output_file}")

if __name__ == '__main__':
    # Check for --no_flip flag
    args = sys.argv[1:]
    flip_angle = 1  # default: flip

    if '--no_flip' in args:
        flip_angle = 0
        args.remove('--no_flip')

    if len(args) < 1:
        print("Usage: python extract_info_from_warpxml.py [--no_flip] input_file1.xml [input_file2.xml ...]")
        sys.exit(1)

    print("!!!! Script to extract information for pytom_match_pick !!!!")
    xmlfiles = args

    for xmlfile in xmlfiles:
        if not xmlfile.endswith('.xml'):
            print(f"Skipping {xmlfile} (not an XML file)")
            continue
        print('--->')

        ts = ET.parse(xmlfile)
        root = ts.getroot()

        def_file = xmlfile.replace('.xml', '_defocus.txt')
        dose_file = xmlfile.replace('.xml', '_dose.txt')
        tlt_file = xmlfile.replace('.xml', '.tlt')

        # Angles
        angle_element = root.find('Angles')
        angle_values = angle_element.text.strip().split()
        angle_values = [float(val) for val in angle_values]

        if flip_angle > 0:
            angle_values = [-val for val in angle_values]
            print("Flip angle to negative (default for McGill data)")

        write_values_to_file(tlt_file, angle_values, label="Angles")

        # Dose
        dose_element = root.find('Dose')
        dose_values = dose_element.text.strip().split()
        write_values_to_file(dose_file, dose_values, label="Dose")

        # Defocus
        z = []
        def_values = []

        gridctf = root.find('GridCTF')
        for node in gridctf.iter('Node'):
            z.append(node.attrib['Z'])

        for zvalue in z:
            nodectf = root.find(f"./GridCTF/Node/[@Z='{zvalue}']")
            nodectfdelta = root.find(f"./GridCTFDefocusDelta/Node/[@Z='{zvalue}']")
            nodeangle = root.find(f"./GridCTFDefocusAngle/Node/[@Z='{zvalue}']")
            defocusv = float(nodectf.attrib['Value'])
            defocusu = float(nodectfdelta.attrib['Value']) + defocusv
            def_values.append((defocusu + defocusv) / 2)

        write_values_to_file(def_file, def_values, label="Defocus")
        
