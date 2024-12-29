#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0.1
Created on Sat Dec  4 22:56:14 2021
Script to downgrade Relion 4.0 star file to Relion 3.0 for Warp subtomo boxing
For now, only deal with original file for import
Usage: relion4_to_warp.py --i run_data.star --o run_data_rln3.0.star --angpix 2.12 --rescale_angpix 8.48
@author: Huy Bui, McGill University
"""


import numpy as np
import pandas as pd
import starfile
import argparse

from eulerangles import euler2matrix

if __name__=='__main__':
	print('Warning!! Only work with simple relion 4.0 particle file')
	parser = argparse.ArgumentParser(description='Convert Relion 4.0 subtomo star to ChimeraX session')
	parser.add_argument('--i', help='Input star file',required=True)
	parser.add_argument('--o', help='Output ChimeraX Script',required=True)
	parser.add_argument('--angpix', help='Pixel size of average',required=True)
	parser.add_argument('--rescale_angpix', help='Box size of average',required=True)

	args = parser.parse_args()
	
	outfile = args.o
	
	rescale_angpix = float(args.rescale_angpix)
	angpix = float(args.angpix)
	

	# Loading Relion 4.0 original star file
	df = starfile.read(args.i)
	#print(df)	
	# Relion 4.0 or 3.1
	
	# Initialize dict       
	data = {}
	data['rlnMicrographName'] = df['rlnTomoName'].astype(str) + ".mrc_{0:.2f}Apx.mrc".format(rescale_angpix)
	data['rlnCoordinateX']=df['rlnCoordinateX'].to_numpy()*angpix/rescale_angpix
	data['rlnCoordinateY']=df['rlnCoordinateY'].to_numpy()*angpix/rescale_angpix
	data['rlnCoordinateZ']=df['rlnCoordinateZ'].to_numpy()*angpix/rescale_angpix
	data['rlnAngleRot']=df['rlnAngleRot'].to_numpy()
	data['rlnAngleTilt']=df['rlnAngleTilt'].to_numpy()
	data['rlnAnglePsi']=df['rlnAnglePsi'].to_numpy()

	# Convert dict to dataframe
	dfout = pd.DataFrame.from_dict(data)
	
	starfile.write(dfout, args.o, overwrite=True)
	print(f"Done! Write '{args.i}' to RELION/Warp compatible STAR file '{args.o}'")

	
