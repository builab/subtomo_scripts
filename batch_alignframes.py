#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created April 2022

Wrapper script to run alignframes (4.11 tested) on a batch of tilt series.
It is possible to use wild card to select a subset of tilt series
Usage: batch_alignframes.py -i TS_01/TS_01.mrc" -frameDir tilt_frames -gainref tilt_frames/CountRef_0.0.mrc -rotation 6
Usage: batch_alignframes.py --i "TS*/TS*.mrc" --frameDir tilt_frames --gainref tilt_frames/CountRef_0.0.mrc --rotation 6

Edit other parameter inside the files
@author: Huy Bui, McGill
"""
import argparse, os, glob


def run_alignframes(tiltseries, mdoc, frameDir, rotation, gainref, vary, gpu):
	"""Run alignframes"""
	cmd = 'alignframes -mdoc ' + mdoc + ' -path ' + frameDir + ' -rotation ' + rotation + ' -vary ' + vary + ' -gain ' + gainref + ' -gpu ' + str(gpu) + ' -normalize ' + tiltseries
	print(cmd)
	os.system(cmd) 


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Batch alignframes of many tilt series\nNeed each TS in a separate folder\nNeed mrc.mdoc file in the same folder')
	parser.add_argument('--i', help='Input Tilt Series',required=True)
	parser.add_argument('--frameDir', help='Directory containing frames',required=True)
	parser.add_argument('--rotation', help='Rotation value (default 6 for McGill Krios K3)',required=False,default="6")
	parser.add_argument('--gainref', help='Gain reference file',required=True)
	parser.add_argument('--gpu', help='Use GPU',required=False,default=0)
	parser.add_argument('--vary', help='Vary value of alignframe',required=False, default="0.1")

  
	args = parser.parse_args()
	
	tsList = glob.glob(args.i)
  
	currentDir = os.getcwd()
  
	for tiltseries in tsList:
		tsName = os.path.basename(tiltseries)
		if not tsName.endswith('.mrc'):
			print('Processing ' + tsName + ': ERROR! Only mrc file supported')
			break
		tsPath = os.path.dirname(tiltseries)
		mdoc = tiltseries.replace('.mrc', '.mrc.mdoc')
		print('Processing ' + tsName)
		run_alignframes(tiltseries, mdoc, args.frameDir, args.rotation, args.gainref, args.vary, args.gpu)

