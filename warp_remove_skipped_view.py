#!/usr/bin/env python3
# Script from https://github.com/hamid13r/warp_remove_skipped_views by hamid13r
# Put in here so we have an integrated script for Bui lab
# Require click & pandas

import os
import shutil
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import io
import click

@click.command()
@click.option('--xml-dir', default='./', help='Directory containing XML files')
@click.option('--xml-pattern', default='*.xml', help='Pattern to match XML files')
@click.option('--backup-dir', default='backup_xml', help='Directory to store XML backups')
@click.option('--tiltstack-dir', default='tiltstack', help='Base directory for tiltstack logs')
@click.option('--all-true', is_flag=True, default=False, help='Set all UseTilt values to True')
@click.option('--n-tilts', default=0, help='Number of tilts by dose to keep and discard the rest')
@click.option('--max-tilt', default=0, help='Maximum tilt angle (calculated from the minimum dose) to keep, others set to False')
def main(xml_dir, xml_pattern, backup_dir, tiltstack_dir, all_true, n_tilts, max_tilt):
    """Process XML files and update UseTilt values based on taSolution.log.
    Args:
        xml_dir (str): Directory containing XML files.
        xml_pattern (str): Pattern to match XML files.
        backup_dir (str): Directory to store XML backups.
        tiltstack_dir (str): Base directory for tiltstack logs.
    Returns:
        None
    
    example
    python remove_skipped_views.py ./ --xml-pattern '*.xml' --backup-dir backup_xml --tiltstack-dir tiltstack
    """
    xml_files = glob.glob(os.path.join(xml_dir, xml_pattern))

    if not xml_files:
        click.echo("No XML files found.")
        return

    os.makedirs(backup_dir, exist_ok=True)

    for xml_file in xml_files:
        backup_path = os.path.join(backup_dir, os.path.basename(xml_file))
        if os.path.exists(backup_path):
            click.echo(f"Backup for {xml_file} already exists, the code will stop to avoid overriding old backups with modified files.")
            return
        else:
            shutil.copy(xml_file, backup_dir)
        # Read corresponding taSolution.log
        log_path = os.path.join(tiltstack_dir, os.path.splitext(os.path.basename(xml_file))[0], "taSolution.log")
        if not os.path.exists(log_path):
            click.echo(f"Log file {log_path} not found, skipping {xml_file}.")
            continue

        with open(log_path, "r") as f:
            lines = f.readlines()

        view_line_idx = next((idx for idx, line in enumerate(lines) if 'view' in line), None)
        if view_line_idx is None:
            click.echo(f"No line with 'view' found in {log_path}, skipping {xml_file}.")
            continue

        data_str = ''.join(lines[view_line_idx:])
        df = pd.read_csv(io.StringIO(data_str), delim_whitespace=True)
        views_in_log = set(df['view'].unique())
        tilts_in_log = df['tilt'].tolist()

        tree = ET.parse(xml_file)
        root = tree.getroot()
        use_tilt_elem = root.find('UseTilt')
        if use_tilt_elem is not None:
            current_values = [val.strip() for val in use_tilt_elem.text.split('\n') if val.strip()]
            updated_values = ['False'] * len(current_values)
            changes_made = 0
            if all_true:
                updated_values = ['True'] * len(current_values)
                changes_made = sum(1 for val in current_values if val != 'True')
            #if n_tilts is 0, set views in log to True, others to False
            elif n_tilts == 0 and max_tilt == 0:
                for i, value in enumerate(current_values):
                    view_number = i + 1
                    if view_number in views_in_log:
                        updated_values[i] = 'True'
                    else:
                        updated_values[i] = 'False'
                        if value == 'True':
                            changes_made += 1
            #if n_tilts > 0, first keep the n_tilts lowest dose views as True 
            elif n_tilts > 0:
                # Read Dose values from XML and sort them
                dose_elems = root.findall('Dose')
                dose_values = [float(elem.text.strip()) for elem in dose_elems]
                sorted_indices = sorted(range(len(dose_values)), key=lambda i: dose_values[i])

                for idx in sorted_indices[:n_tilts]:
                    updated_values[idx] = 'True'
                for idx in sorted_indices[n_tilts:]:
                    updated_values[idx] = 'False'
                changes_made = sum(1 for i, val in enumerate(current_values) if updated_values[i] != val)
                # Now ensure views not in log are False
                for i, value in enumerate(current_values):
                    view_number = i + 1
                    if view_number in views_in_log:
                        updated_values[i] = 'True'
                    else:
                        updated_values[i] = 'False' 
                        if value == 'True':
                            changes_made += 1
            elif max_tilt > 0:
                #first find the minimum dose tilt
                dose_elems = root.findall('Dose')
                dose_values = [float(elem.text.strip()) for elem in dose_elems]
                min_dose_idx = dose_values.index(min(dose_values))
                min_dose_tilt = tilts_in_log[min_dose_idx]
                #calculate the maximum tilt to keep
                max_tilt_to_keep = abs(min_dose_tilt) + max_tilt
                for i, value in enumerate(current_values):
                    view_number = i + 1
                    tilt_angle = tilts_in_log[i]
                    if view_number in views_in_log and abs(tilt_angle) <= max_tilt_to_keep:
                        updated_values[i] = 'True'
                    else:
                        updated_values[i] = 'False'
                        if value == 'True':
                            changes_made += 1


            use_tilt_elem.text = '\n'.join(updated_values)
            click.echo(f"{xml_file}: {changes_made} changes made to UseTilt.")

        tree.write(xml_file)

if __name__ == '__main__':
    main()