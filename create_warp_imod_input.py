#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 2022 by Huy Bui, McGill University
Generate IMOD folder structure for WARP
You need to run in the main folder of the tilt series.
The script will look for the mdoc file under the regular expression enter
python create_warp_imod_input.py [tilt_series_pattern]
E.g. python create_warp_imod_input.py "FAP256*"
@author: kbui2
"""


import glob, sys
import os, shutil

if __name__=='__main__':

	pattern = sys.argv[1]
	
	warpDir = 'warp_imod'
	
	mdocfiles = glob.glob(pattern + '/*.mrc.mdoc')
	
	if len(mdocfiles) < 1:
		print('No folder is found with pattern: ' + pattern)
	
	try:
		os.mkdir(warpDir)
	except OSError as exc:
		if exc.errno != errno.EEXIST:
			raise
		pass

	for mdoc in mdocfiles:
		tsName = os.path.basename(mdoc).replace('.mrc.mdoc', '')
		tsPath = os.path.dirname(mdoc)
		print('Processing tilt series ' + tsName)
		
		# Mkdir folder name
		destDir = warpDir + '/' + tsName + '.mrc'
		try:
			os.mkdir(destDir)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise
			pass
		
		# Copy taSolution
		try:
			shutil.copy(tsPath + '/taSolution.log', destDir)
			print("\ttaSolution.log copied successfully.")
			shutil.copy(tsPath + '/' + tsName + '.xf', destDir + '/' + tsName + '.mrc.xf')
			print('\t' + tsName + '.xf copied successfully!')		
		except PermissionError:
			print("Permission denied.")
		# For other errors
		except:
			print("Error occurred while copying file.")
			