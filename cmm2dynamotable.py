#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:33:47 2021
Fixed Dec 13, non-continous subtomo number using np.asis

Trim the Dynamo table based on a Chimera cmm file
Using dynamo_table2chimeramarker to generate the cmm file from 1 tomogram
Load the cmm & tomo in Chimera
Delete unnecessary particles on carbon etc
Save the cmm as a new file
Generate new table

python cmm2dynamotable.py --i table.tbl --o table_out.tbl --cmm sel_markers.cmm

@author: kbui2
"""

import numpy as np
import argparse
import xml.etree.ElementTree as ET

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Write new table based on selected Cmm file')
	parser.add_argument('--i', help='Input table file',required=True)
	parser.add_argument('--o', help='Output table file',required=True)
	parser.add_argument('--cmm', help='Chimera cmm file',required=True)

	args = parser.parse_args()

	intable = np.genfromtxt(args.i, delimiter=' ')
	markerset = ET.parse(args.cmm)
	root = markerset.getroot()
	
	listpar = []
	# Obtain all marker id
	for marker in root.iter('marker'):
		listpar.append(int(marker.attrib['id']))
		
	#listnp = np.array(listpar)
	outtable=intable[np.isin(intable[:, 0], listpar)]
	#print(intable[:, 0])
	#print(listpar)
	print('\nWriting {:d} particles to {:s}!'.format(len(listpar), args.o))
	np.savetxt(args.o, outtable, fmt='%.2f', delimiter=' ')
	
