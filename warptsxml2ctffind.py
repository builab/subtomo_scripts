#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2022 by Huy Bui, McGill University
Extract ctf info from the tilt series .mrc.xml file into ctffind4 format
python warptsxml2ctffind.py
@author: kbui2
"""

import numpy as np
import xml.etree.ElementTree as ET
import glob

if __name__=='__main__':
	xmlfiles = glob.glob('*.xml')

	for xmlfile in xmlfiles:
		ts = ET.parse(xmlfile)
		outfile = xmlfile.replace('.xml', '_output.txt')
		root = ts.getroot()
		
		z = []
		data = []
		ctfResEst = float(root.attrib["CTFResolutionEstimate"])
	
		gridctf = root.find('GridCTF')
		for node in gridctf.iter('Node'):
			z.append(node.attrib['Z'])
				
		for zvalue in z:
			nodectf = root.find(f"./GridCTF/Node/[@Z='{zvalue}']")
			nodectfdelta = root.find(f"./GridCTFDefocusDelta/Node/[@Z='{zvalue}']")
			nodeangle = root.find(f"./GridCTFDefocusAngle/Node/[@Z='{zvalue}']")
			defocusv = float(nodectf.attrib['Value'])*10000
			defocusu = float(nodectfdelta.attrib['Value'])*10000 + defocusv
			angle = float(nodeangle.attrib['Value'])
			data.append([float(zvalue) + 1, defocusu, defocusv, angle, 0, 0, ctfResEst])

		np.savetxt(outfile, data, fmt='%.2f', delimiter=' ')
	
