#!/usr/bin/env python3
# Script to use fidder to erase the fiducial mask
# Code from Jerry Gao, dx.doi.org/10.17504/protocols.io.6qpvr8qbblmk/v3, ChatGPT, Huy Bui
# Separate mask & prediction folder so we can use the mask to correct for even/odd separately from average

import os
import mrcfile
import torch
from multiprocessing import Pool
import argparse
import numpy as np
import glob
from fidder.predict import predict_fiducial_mask
from fidder.erase import erase_masked_region


def erase_gold(filename, input_dir, mask_dir, norename):
	"""Apply fidder's erase_masked_region function to a single frame and save the result as a new mrc file.
		
	Args:
		filename (str): The name of the micrograph to process.
		input_dir (str) :
		mask_dir (str) : 
	"""
	mic_path = os.path.join(input_dir, filename)
	mask_path = os.path.join(mask_dir, filename)

	image = torch.tensor(mrcfile.read(mic_path))
	if not os.path.exists(mask_path):
		print(f"Error: Mask path '{mask_path}' does not exist. Skip!!")
		return
	mask = torch.tensor(mrcfile.read(mask_path))

	erased_image = erase_masked_region(image=image, mask=mask)

	output_filename = f"{os.path.splitext(filename)[0]}_erased.mrc"
	mrc_output_path = os.path.join(input_dir, output_filename)
	with mrcfile.new(mrc_output_path, overwrite=True) as mrc:
		mrc.set_data(erased_image.numpy())

	print('Write' + output_filename + ' completed')
	rename_files(mic_path, mrc_output_path, norename)
	
def rename_files(file, erased_file, norename):
	"""
	Renames files based on the provided condition.

	Args:
		file (str): The original file path.
		erased_file (str): The file to be renamed to the original file path.
		norename (bool): If True, no renaming occurs. Default is False.

	Returns:
		None
	"""
	if not norename:
		backup_file = f"{file}~"
		try:
			# Rename the original file to the backup file
			shutil.move(file, backup_file)
			print(f"Moved {file} to {backup_file}")
			
			# Rename the erased file to the original file name
			shutil.move(erased_file, file)
			print(f"Moved {erased_file} to {file}")
		except Exception as e:
			print(f"Error during file renaming: {e}")
 
def main():
	parser = argparse.ArgumentParser(description="Process MRC files in a directory with specified parameters.")
	parser.add_argument('--idir', required=True, help="Input directory containing .mrc files.")
	parser.add_argument('--mdir', required=True, help="Mask directory to store output .mrc files.")
	parser.add_argument('--norename', action='store_true', help="Don't rename output file to the same as input file.")
	parser.add_argument('--j', type=int, required=True, default=20, help="Number of threads")


	args = parser.parse_args()

	# Validate input directory
	if not os.path.isdir(args.idir):
		raise ValueError(f"Input directory {args.idir} does not exist.")

	# Validate mask directory
	if not os.path.isdir(args.mdir):
		os.makedirs(args.mdir, exist_ok=True)

	# Find all .mrc files in the input directory
	mrc_files = [f for f in os.listdir(args.idir) if f.endswith('.mrc')]
	if not mrc_files:
		print("No .mrc files found in the input directory.")
		return

	# Process each .mrc file
	with Pool(args.j) as p:
		p.starmap(erase_gold, [(mrc_file, args.idir, args.mdir, args.norename) for mrc_file in mrc_files])
	print('################################ all gold erased for ' + args.idir + ' ################################')

if __name__ == '__main__':
	main()
