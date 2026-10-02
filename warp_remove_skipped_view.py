#!/usr/bin/env python3
# Script converted from click to argparse
# Original script from https://github.com/hamid13r/warp_remove_skipped_views by hamid13r
# Put in here so we have an integrated script for Bui lab
# Require pandas
# Use argparse instead of click
#
# Modification: XML files for which no UseTilt value would change are left
# completely untouched (no backup, no rewrite, mtime preserved). Only files
# that actually need modification are backed up and rewritten.

import os
import shutil
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import io
import argparse
import sys  # Import sys for printing messages/errors


def main():
    """Process XML files and update UseTilt values based on taSolution.log.

    This function sets up the argument parser, parses arguments, and
    orchestrates the file processing logic.
    """
    parser = argparse.ArgumentParser(
        description='Process XML files and update UseTilt values based on taSolution.log.',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--xml-dir',
        default='./',
        help='Directory containing XML files (default: ./)')

    parser.add_argument(
        '--xml-pattern',
        default='*.xml',
        help='Pattern to match XML files (default: *.xml)')

    parser.add_argument(
        '--backup-dir',
        default='backup_xml',
        help='Directory to store XML backups (default: backup_xml)')

    parser.add_argument(
        '--tiltstack-dir',
        default='tiltstack',
        help='Base directory for tiltstack logs (default: tiltstack)')

    parser.add_argument(
        '--all-true',
        action='store_true',
        default=False,
        help='Set all UseTilt values to True (default: False)')

    parser.add_argument(
        '--n-tilts',
        type=int,
        default=0,
        help='Number of tilts by dose to keep and discard the rest (default: 0)')

    parser.add_argument(
        '--max-tilt',
        type=float,
        default=0,
        help='Maximum tilt angle (calculated from the minimum dose) to keep, others set to False (default: 0)')

    args = parser.parse_args()

    process_xml_files(
        args.xml_dir,
        args.xml_pattern,
        args.backup_dir,
        args.tiltstack_dir,
        args.all_true,
        args.n_tilts,
        args.max_tilt
    )


def process_xml_files(xml_dir, xml_pattern, backup_dir, tiltstack_dir, all_true, n_tilts, max_tilt):
    """Core logic to process XML files and update UseTilt values.

    Files where no UseTilt value changes are left untouched. Files that do
    need changes are backed up to backup_dir before being rewritten.

    Args:
        xml_dir (str): Directory containing XML files.
        xml_pattern (str): Pattern to match XML files.
        backup_dir (str): Directory to store XML backups.
        tiltstack_dir (str): Base directory for tiltstack logs.
        all_true (bool): Set all UseTilt values to True.
        n_tilts (int): Number of tilts by dose to keep.
        max_tilt (float): Maximum tilt angle to keep.
    """
    xml_files = glob.glob(os.path.join(xml_dir, xml_pattern))

    if not xml_files:
        print("No XML files found.", file=sys.stderr)
        return

    for xml_file in xml_files:
        # Read corresponding taSolution.log
        log_path = os.path.join(tiltstack_dir, os.path.splitext(os.path.basename(xml_file))[0], "taSolution.log")

        if not os.path.exists(log_path):
            print(f"Log file {log_path} not found, skipping {xml_file}.", file=sys.stderr)
            continue

        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
        except IOError:
            print(f"Could not read log file {log_path}, skipping {xml_file}.", file=sys.stderr)
            continue

        view_line_idx = next((idx for idx, line in enumerate(lines) if 'view' in line), None)
        if view_line_idx is None:
            print(f"No line with 'view' found in {log_path}, skipping {xml_file}.", file=sys.stderr)
            continue

        data_str = ''.join(lines[view_line_idx:])

        try:
            df = pd.read_csv(io.StringIO(data_str), sep=r'\s+')
        except pd.errors.ParserError:
            print(f"Could not parse data from {log_path}, skipping {xml_file}.", file=sys.stderr)
            continue

        # Check if 'view' column exists before trying to access it
        if 'view' not in df.columns:
            print(f"The 'view' column was not found in {log_path}, skipping {xml_file}.", file=sys.stderr)
            continue

        views_in_log = set(df['view'].unique())
        tilts_in_log = df['tilt'].tolist()

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except ET.ParseError:
            print(f"Could not parse XML file {xml_file}, skipping.", file=sys.stderr)
            continue

        use_tilt_elem = root.find('UseTilt')

        if use_tilt_elem is None:
            print(f"Warning: <UseTilt> element not found in {xml_file}.", file=sys.stderr)
            continue

        current_values = [val.strip() for val in use_tilt_elem.text.split('\n') if val.strip()]
        updated_values = ['False'] * len(current_values)
        changes_made = 0

        # --- Argument Logic ---
        if all_true:
            updated_values = ['True'] * len(current_values)
            changes_made = sum(1 for val in current_values if val != 'True')

        # n_tilts > 0 logic
        elif n_tilts > 0:
            # Read Dose values from XML and sort them
            dose_elems = root.findall('Dose')
            dose_values = [float(elem.text.strip()) for elem in dose_elems]

            if len(dose_values) != len(current_values):
                print(f"Dose element count does not match UseTilt count in {xml_file}, skipping.", file=sys.stderr)
                continue

            # Sort indices by dose value to find the n_tilts lowest dose views
            sorted_indices = sorted(range(len(dose_values)), key=lambda i: dose_values[i])

            for idx in sorted_indices[:n_tilts]:
                updated_values[idx] = 'True'
            for idx in sorted_indices[n_tilts:]:
                updated_values[idx] = 'False'

            # A view is only 'True' if it is one of the n_tilts lowest dose
            # views AND it is present in the log (i.e. not a skipped view).
            final_values = []
            for i, value in enumerate(updated_values):
                view_number = i + 1
                if value == 'True' and view_number in views_in_log:
                    final_values.append('True')
                else:
                    final_values.append('False')

            updated_values = final_values
            changes_made = sum(1 for i, val in enumerate(current_values) if updated_values[i] != val)

        # max_tilt > 0 logic
        elif max_tilt > 0:
            dose_elems = root.findall('Dose')
            dose_values = [float(elem.text.strip()) for elem in dose_elems]

            if len(dose_values) != len(current_values) or len(tilts_in_log) != len(current_values):
                print(f"Element count mismatch between Dose/Tilts/UseTilt in {xml_file}, skipping.", file=sys.stderr)
                continue

            # Find the tilt angle for the minimum dose view
            min_dose_idx = dose_values.index(min(dose_values))
            min_dose_tilt = tilts_in_log[min_dose_idx]

            # Calculate the maximum absolute tilt angle to keep
            max_tilt_to_keep = abs(min_dose_tilt) + max_tilt

            for i, value in enumerate(current_values):
                view_number = i + 1
                tilt_angle = tilts_in_log[i]

                # Keep if in log AND within max_tilt_to_keep range
                if view_number in views_in_log and abs(tilt_angle) <= max_tilt_to_keep:
                    updated_values[i] = 'True'
                else:
                    updated_values[i] = 'False'

            changes_made = sum(1 for i, val in enumerate(current_values) if updated_values[i] != val)

        # Default logic: n_tilts == 0 and max_tilt == 0 -> just filter by views_in_log
        else:
            for i, value in enumerate(current_values):
                view_number = i + 1
                if view_number in views_in_log:
                    updated_values[i] = 'True'
                else:
                    updated_values[i] = 'False'

            changes_made = sum(1 for i, val in enumerate(current_values) if updated_values[i] != val)

        # --- Leave file untouched if nothing changes ---
        if changes_made == 0:
            print(f"{xml_file}: no changes needed, leaving file untouched.")
            continue

        # --- Back up only files we are actually going to modify ---
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, os.path.basename(xml_file))

        # Check for existing backup and prevent override
        if os.path.exists(backup_path):
            print(f"Backup for {xml_file} already exists, the code will stop to avoid overriding old backups with modified files.", file=sys.stderr)
            return
        shutil.copy(xml_file, backup_path)

        # Apply and write changes
        use_tilt_elem.text = '\n'.join([''] + updated_values + [''])  # Add newlines for formatting
        tree.write(xml_file)
        print(f"{xml_file}: {changes_made} changes made to UseTilt.")


if __name__ == '__main__':
    main()