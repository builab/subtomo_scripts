#!/usr/bin/env python3
# Script to use fidder to predict the fiducial mask and also add dead pixel mask to the mask
# Code from Jerry Gao, dx.doi.org/10.17504/protocols.io.6qpvr8qbblmk/v3, ChatGPT, Huy Bui
# Separate mask & prediction folder so we can use the mask to correct for even/odd separately

import os
import mrcfile
import torch
import argparse
import numpy as np
import glob
from fidder.predict import predict_fiducial_mask

def draw_circle(array, center_x, center_y, radius):
	height, width = array.shape
	y_indices, x_indices = np.ogrid[:height, :width]
	mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
	array[mask] = 1
	
def write_indices_to_txt(mask, output_file):
	"""
	Write the 1-based indices of pixels with value 1 in a numpy array to a .txt file.

	Args:
		mask (numpy.ndarray): The input 2D numpy array containing values 0 and 1.
		output_file (str): Path to the output .txt file.
	"""
	# Find the indices of pixels with value 1
	y_indices, x_indices = np.where(mask == 1)
	
	# Convert to 1-based indexing
	x_indices += 1
	y_indices += 1

	# Write to the output file
	with open(output_file, 'w') as f:
		for x, y in zip(x_indices, y_indices):
			f.write(f"{x} {y}\n")
	print(f"Mask written to {output_file}")
	

def make_mask(filename, input_dir, mask_dir, angpix, thresh, coords_file, deadpixel_radius, ignore_existing, use_txt):
	""" Apply fidder's predict_fiducial_mask function to a single micrograph and save the resultant mask as a new mrc file.
	Args:
		filename (str): The name of the micrograph to process.
		input_dir (str): The directory containing micrographs.
		mask_dir (str): The directory to store generated masks.
		angpix (float): Pixel size in angstroms.
		thresh (float): Probability threshold for fiducial detection.
		coords_file (str): Path to the file with dead pixel coordinates.
		deadpixel_radius (int): Radius for masking dead pixels.
		ignore_existing (bool): Skip files that already have masks in the mask directory.
		use_txt (bool): Not writing .mrc file.

	"""
	mic_path = os.path.join(input_dir, filename)
	mask_path = os.path.join(mask_dir, filename)

	if ignore_existing and os.path.exists(mask_path):
		print(f"Skipping {filename}: mask already exists.")
		return

	image = torch.tensor(mrcfile.read(mic_path))

	mask, probabilities = predict_fiducial_mask(
		image, pixel_spacing=angpix, probability_threshold=thresh
	)

	mask_uint8 = mask.to(torch.uint8)

	os.makedirs(os.path.dirname(mask_path), exist_ok=True)

	# Add deadpixel mask
	if coords_file:
		with open(coords_file, 'r') as f:
			for line in f:
				parts = line.split()
				if len(parts) < 2:
					continue
				x_coord = float(parts[0])
				y_coord = float(parts[1])
				draw_circle(mask_uint8, x_coord, y_coord, deadpixel_radius)



	# Write mask to txt file
	output_txt = f"{os.path.splitext(filename)[0]}.txt"
	write_indices_to_txt(mask_uint8.numpy(), os.path.join(mask_dir, output_txt))

	if not use_txt:
		with mrcfile.new(mask_path, overwrite=True) as mrc:
			mrc.set_data(mask_uint8.numpy())
		print(f"Mask created for {filename}.")

def main():
	parser = argparse.ArgumentParser(description="Process MRC files in a directory with specified parameters.")
	parser.add_argument('--idir', required=True, help="Input directory containing .mrc files.")
	parser.add_argument('--mdir', required=True, help="Mask directory to store output .mrc files.")
	parser.add_argument('--angpix', required=True, type=float, help="Pixel size.")
	parser.add_argument('--p', type=float, default=0.95, help="Threshold parameter for predicting fiducial (default: 0.95).")
	parser.add_argument('--deadpix_file', required=False, default="", help="Text file containing dead pixel coordinates.")
	parser.add_argument('--deadpix_radius', type=int, default=3, help="Radius around dead pixels to mask (default: 3).")
	parser.add_argument('--ignore_existing', action='store_true', help="Skip files that already have masks in the mask directory.")
	parser.add_argument('--use_txt', action='store_true', help="Only write coordinates, not mrc mask files")

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
	for mrc_file in mrc_files:
		make_mask(mrc_file, args.idir, args.mdir, args.angpix, args.p, args.deadpix_file, args.deadpix_radius, args.ignore_existing, args.use_txt)

if __name__ == '__main__':
	main()
