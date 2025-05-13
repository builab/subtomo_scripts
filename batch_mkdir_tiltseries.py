#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8, 2021

All .mrc & mrc.mdoc, log file should be in 1 folder
Move .mrc file into folders

@author: Huy Bui, McGill
"""
import argparse, os, glob, shutil

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Organize tilt series into folders\n. E.g.: batch_mkdir_tiltseries.py --i \"CU482_0*.mrc\"')
	parser.add_argument('--i', help='Input Tilt Series Wild Card',required=True)
	
	args = parser.parse_args()
	tsList = glob.glob(args.i)
	for tiltseries in tsList:
		if not tiltseries.endswith('.mrc'):
			print('Processing ' + tiltseries + ': ERROR! Only mrc file supported')
			break
		tsName = tiltseries.replace('.mrc', '')
		mrcDoc = tiltseries.replace('.mrc', '.mrc.mdoc')
		log = tiltseries.replace('.mrc', '.log')

		try:
			os.mkdir(tsName)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise
			pass
		# Move tilt series
		print('mv ' + tiltseries + ' ' + os.path.join(tsName, tiltseries))
		shutil.move(tiltseries, os.path.join(tsName, tiltseries))
		print('mv ' + mrcDoc + ' ' + os.path.join(tsName, mrcDoc))
		shutil.move(mrcDoc, os.path.join(tsName, mrcDoc))
		print('mv ' + log + ' ' + os.path.join(tsName, mrcDoc))
		shutil.move(log, os.path.join(tsName, log))



