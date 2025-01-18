#!/usr/bin/env python3
# Script to use fidder to erase the fiducial mask
# Code from Jerry Gao, dx.doi.org/10.17504/protocols.io.6qpvr8qbblmk/v3, ChatGPT, Huy Bui
# Separate mask & prediction folder so we can use the mask to correct for even/odd separately from average

import os
import shutil
import mrcfile
import torch
from multiprocessing import Pool
import argparse
import numpy as np
import glob
from fidder.predict import predict_fiducial_mask
from fidder.erase import erase_masked_region


def erase_gold(filename, input_dir, mask_dir, norename, use_coord, xdim, ydim):
	"""Apply fidder's erase_masked_region function to a single frame and save the result as a new mrc file.
		
	Args:
		filename (str): The name of the micrograph to process.
		input_dir (str) :
		mask_dir (str) : 
	"""
	
	writeMask = True # Set True for debugging, set False for operation
	mic_path = os.path.join(input_dir, filename)
	
	# Check for using txt file instead of mrc file
	if use_coord: 
		mask_path = os.path.join(mask_dir, 	f"{os.path.splitext(filename)[0]}.txt")
		if not os.path.exists(mask_path):
			print(f"Error: Mask path '{mask_path}' does not exist. Skip!!")
			return
		mask = torch.tensor(read_coordinates_to_mask(mask_path, xdim, ydim))
		# Only for debugging
		if writeMask:
			mrc_mask_path = os.path.join(mask_dir, filename)
			with mrcfile.new(mrc_mask_path, overwrite=True) as mrc:
				mrc.set_data(mask_uint8.numpy())
	else:
		mask_path = os.path.join(mask_dir, filename)
		if not os.path.exists(mask_path):
			print(f"Error: Mask path '{mask_path}' does not exist. Skip!!")
			return
		mask = torch.tensor(mrcfile.read(mask_path))
		
	image = torch.tensor(mrcfile.read(mic_path))


	erased_image = erase_masked_region(image=image, mask=mask)

	output_filename = f"{os.path.splitext(filename)[0]}_erased.mrc"
	mrc_output_path = os.path.join(input_dir, output_filename)
	with mrcfile.new(mrc_output_path, overwrite=True) as mrc:
		mrc.set_data(erased_image.numpy())

	print('Write' + output_filename + ' completed')
	rename_files(mic_path, mrc_output_path, norename)

def erase_gold_old(filename, input_dir, mask_dir, norename):
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
	
def read_coordinates_to_mask(coord_file, xdim, ydim):
	"""
	Read a coordinate file and create a numpy array of size (xdim, ydim) with values set to 1
	at the given coordinates and 0 elsewhere.

	Args:
		coord_file (str): Path to the coordinate file.
						 Each line in the file should contain "X Y" (1-based indexing).
		xdim (int): Width of the output array.
		ydim (int): Height of the output array.

	Returns:
		numpy.ndarray: A 2D numpy array of shape (ydim, xdim) with the specified values.
	"""
	# Initialize an array filled with zeros
	array = np.zeros((ydim, xdim), dtype=np.uint8)

	# Read the coordinates from the file
	with open(coord_file, 'r') as f:
		for line in f:
			# Parse the coordinates (convert from 1-based to 0-based indexing)
			parts = line.strip().split()
			if len(parts) != 2:
				continue  # Skip invalid lines
			x, y = map(int, parts)
			array[y - 1, x - 1] = 1  # Convert to 0-based indexing and set to 1

	return array
	
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
	parser.add_argument('--j', type=int, default=20, help="Number of threads")
	parser.add_argument('--xdim', type=int, default=5760, help="Micrograph X dimension (default 5760)")
	parser.add_argument('--ydim', type=int, default=4092, help="Micrograph Y dimension (default 4092)")
	parser.add_argument('--use_coord', action='store_true', help="Use coordinate txt file, not mrc.")



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
		p.starmap(erase_gold, [(mrc_file, args.idir, args.mdir, args.norename, args.use_coord, args.xdim, args.ydim) for mrc_file in mrc_files])
	print('################################ all gold erased for ' + args.idir + ' ################################')

if __name__ == '__main__':
	main()
