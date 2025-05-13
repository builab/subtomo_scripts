#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert eman3d file to cmm for quick checking
Also, option for separate color
TODO: Better to use ElementTree
HB, McGill, 2021
"""
import pandas as pd
import argparse
import xml.etree.ElementTree as ET

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Write Chimera CMM file from EMAN3D coordinate file')
	parser.add_argument('--i', help='Input EMAN 3D coordinate',required=True)
	parser.add_argument('--o', help='Output Chimera File',required=True)
	parser.add_argument('--bin', help='Bin factor of EMAN Coordinate',required=True)
	parser.add_argument('--radius', help='Radius of marker',required=False, default=5)


	args = parser.parse_args()
  
	binFactor = float(args.bin)
	header_list = ["X", "Y", "Z"]
	radius = int(args.radius)
	
	out = open(args.o, 'w')
	df = pd.read_csv(args.i, delim_whitespace=True, names=header_list, index_col=False)
	
	out.write("<marker_set name=\"marker set 1\">\n")
	for i in range(len(df['X'])):
		x = float(df.loc[i, 'X'])*binFactor
		y = float(df.loc[i, 'Y'])*binFactor
		z = float(df.loc[i, 'Z'])*binFactor
		out.write("<marker id=\"{:d}\" x=\"{:.1f}\" y=\"{:.1f}\" z=\"{:.1f}\" radius=\"{:d}\"/>\n".format(i + 1, x, y, z, radius))
		
	out.write("</marker_set>\n")
	out.close()
	
	
