#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 8 2021
Script to run batch imod through the command line using a template from one reconstructed tomogram
v0.7 April 2021 Use exclude views from align.com and tilt.com
TODO: NO TILT SIRT (not really important)
TODO: implement multiprocessing for cluster submission
NOT WORKING YET

is is used specifically for K3 in McGill. Imod 4.11.8

@author: Huy Bui, McGill
"""

import argparse, os, glob, shutil
from multiprocessing import Pool


	
def run_eraser(baseName, tempCont):
	''' Write eraser and run'''
	outCom = open(operation + '.com', 'w')
	for line in tempCont:
		if line.startswith('InputFile'):
			outCom.write('InputFile\t{:s}.mrc\n'.format(baseName))
		elif line.startswith('OutputFile'):
			outCom.write('OutputFile\t{:s}_fixed.mrc\n'.format(baseName))
		elif line.startswith('PointModel'):
			outCom.write('PointModel\t{:s}_peak.mod\n'.format(baseName))
		#elif line.startswith('BinByFactor'):
		#	outCom.write('BinByFactor\t{:d}.xf\n'.format(binFactor)
		else:
			outCom.write(line)
	outCom.close()
	
def run_prenewst(baseName, tempCont):
	'''Write prenewst and run'''
	outCom = open(operation + '.com', 'w')
	for line in tempCont:
		if line.startswith('InputFile'):
			outCom.write('InputFile\t{:s}.mrc\n'.format(baseName))
		elif line.startswith('OutputFile'):
			outCom.write('OutputFile\t{:s}_preali.mrc\n'.format(baseName))
		elif line.startswith('TransformFile'):
			outCom.write('TransformFile\t{:s}.prexg\n'.format(baseName))	
		elif line.endswith('.prexf\n'):
			outCom.write('{:s}.prexf\n'.format(baseName))	
		elif line.endswith('.prexg\n'):
			outCom.write('{:s}.prexg\n'.format(baseName))	
		else:
			outCom.write(line)
	outCom.close()

def run_newst(baseName, tempCont):
	'''Write newst and run'''
	outCom = open(operation + '.com', 'w')
	for line in tempCont:
		if line.startswith('InputFile'):
			outCom.write('InputFile\t{:s}.mrc\n'.format(baseName))
		elif line.startswith('OutputFile'):
			outCom.write('OutputFile\t{:s}_ali.mrc\n'.format(baseName))
		elif line.startswith('TransformFile'):
			outCom.write('TransformFile\t{:s}.xf\n'.format(baseName))
		else:
			outCom.write(line)
	outCom.close()
		
def run_align(baseName, tempCont, excludeList):
	'''Write align and run'''
	outCom = open(operation + '.com', 'w')
	for line in tempCont:
		if line.startswith('ModelFile'):
			outCom.write('ModelFile\t{:s}.fid\n'.format(baseName))
		elif line.startswith('ImageFile'):
			outCom.write('ImageFile\t{:s}_preali.mrc\n'.format(baseName))
		elif line.startswith('OutputModelFile'):
			outCom.write('OutputModelFile\t{:s}.3dmod\n'.format(baseName))
		elif line.startswith('OutputResidualFile'):
			outCom.write('OutputResidualFile\t{:s}.resid\n'.format(baseName))
		elif line.startswith('OutputFidXYZFile'):
			outCom.write('OutputFidXYZFile\t{:s}fid.xyz\n'.format(baseName))
		elif line.startswith('OutputTiltFile'):
			outCom.write('OutputTiltFile\t{:s}.tlt\n'.format(baseName))
		elif line.startswith('OutputXAxisTiltFile'):
			outCom.write('OutputXAxisTiltFile\t{:s}.xtilt\n'.format(baseName))
		elif line.startswith('OutputTransformFile'):
			outCom.write('OutputTransformFile\t{:s}.tltxf\n'.format(baseName))
		elif line.startswith('OutputFilledInModel'):
			outCom.write('OutputFilledInModel\t{:s}_nogap.fid\n'.format(baseName))
		elif line.startswith('TiltFile'):
			outCom.write('TiltFile\t{:s}.rawtlt\n'.format(baseName))
			if excludeList != "":
				outCom.write('ExcludeList\t{:s}\n'.format(excludeList))
		elif line.startswith('InputFile1'):
			outCom.write('InputFile1\t{:s}.prexg\n'.format(baseName))
		elif line.startswith('InputFile2'):
			outCom.write('InputFile2\t{:s}.tltxf\n'.format(baseName))
		elif line.startswith('OutputFile'):
			outCom.write('OutputFile\t{:s}_fid.xf\n'.format(baseName))
		elif line.startswith('$b3dcopy -p'):
			outCom.write('$b3dcopy -p {:s}_fid.xf {:s}.xf\n'.format(baseName, baseName))	
			outCom.write('$b3dcopy -p {:s}.tlt {:s}_fid.tlt\n'.format(baseName, baseName))			
		elif 'patch2imod' in line:
			outCom.write('$if (-e {:s}.resid) patch2imod -s 10 {:s}.resid {:s}.resmod\n'.format(baseName, baseName, baseName))
		else:
			outCom.write(line)
	outCom.close()
				     
def run_tilt(baseName, tempCont, excludeList):
	outCom = open('tilt.com', 'w')	
	
				     
	for line in tempCont:
		if line.startswith('InputProjections'):
			outCom.write('InputProjections\t{:s}_ali.mrc\n'.format(baseName))
		elif line.startswith('OutputFile'):
			outCom.write('OutputFile\t{:s}_full_rec.mrc\n'.format(baseName))
		elif line.startswith('TILTFILE'):
			outCom.write('TILTFILE\t{:s}.tlt\n'.format(baseName))
			if excludeList != "":
				outCom.write('EXCLUDELIST\t{:s}\n'.format(excludeList))
		#elif line.startswith('THICKNESS'):
		#	outCom.write('THICKNESS\t{:s}\n'.format(thickness)
		elif line.startswith('XTILTFILE'):
			outCom.write('XTILTFILE\t{:s}.xtilt\n'.format(baseName))
		else:
			outCom.write(line)
	outCom.close()	
	
				     
#def run_tilt_for_sirt(baseName, tempCont):
	# Not doing anything yet
		
def run_newst_3dfind(baseName, tempCont):
	outCom = open('newst_3dfind.com', 'w')	
				     
	for line in tempCont:
		if line.startswith('InputFile'):
			outCom.write('InputFile\t{:s}.mrc\n'.format(baseName))
		elif line.startswith('OutputFile'):
			outCom.write('OutputFile\t{:s}_3dfind_ali.mrc\n'.format(baseName))
		elif line.startswith('TransformFile'):
			outCom.write('TransformFile\t{:s}.xf\n'.format(baseName))
		else:
			outCom.write(line)
	outCom.close()	
				     
def run_tilt_3dfind(baseName, tempCont, excludeList):
	outCom = open('tilt_3dfind.com', 'w')	
				     
	for line in tempCont:
		if line.startswith('InputProjections'):
			outCom.write('InputProjections\t{:s}_3dfind_ali.mrc\n'.format(baseName))
		elif line.startswith('OutputFile'):
			outCom.write('OutputFile\t{:s}_3dfind_rec.mrc\n'.format(baseName))
		elif line.startswith('TILTFILE'):
			outCom.write('TILTFILE\t{:s}.tlt\n'.format(baseName))
			if excludeList != "":
				outCom.write('ExcludeList\t{:s}\n'.format(excludeList))
		#elif line.startswith('THICKNESS'):
		#	outCom.write('THICKNESS\t{:s}\n'.format(thickness)
		elif line.startswith('XTILTFILE'):
			outCom.write('XTILTFILE\t{:s}.xtilt\n'.format(baseName))
		else:
			outCom.write(line)
	outCom.close()		     
				    
def run_findbeads3d(baseName, tempCont):
	outCom = open('findbeads3d.com', 'w')	
				     
	for line in tempCont:
		if line.startswith('InputFile'):
			outCom.write('InputFile\t{:s}_3dfind_rec.mrc\n'.format(baseName))
		elif line.startswith('OutputFile'):
			outCom.write('OutputFile\t{:s}_3dfind.mod\n'.format(baseName))
		else:
			outCom.write(line)
	outCom.close()	

def run_tilt_3dfind_reproject(baseName, tempCont):
	outCom = open('tilt_3dfind_reproject.com', 'w')	
				     
	for line in tempCont:
		if line.startswith('InputProjections'):
			outCom.write('InputProjections\t{:s}_3dfind_ali.mrc\n'.format(baseName))
		elif line.startswith('OutputFile'):
			outCom.write('OutputFile\t{:s}_erase.fid\n'.format(baseName))
		elif line.startswith('TILTFILE'):
			outCom.write('TILTFILE\t{:s}.tlt\n'.format(baseName))
		elif line.startswith('XTILTFILE'):
			outCom.write('XTILTFILE\t{:s}.xtilt\n'.format(baseName))
		elif line.startswith('ProjectModel'):
			outCom.write('ProjectModel\t{:s}_3dfind.mod\n'.format(baseName))
		else:
			outCom.write(line)
	outCom.close()	
				     
def run_golderaser(baseName, tempCont):
	outCom = open('golderaser.com', 'w')	
				     
	for line in tempCont:
		if line.startswith('InputFile'):
			outCom.write('InputFile\t{:s}_ali.mrc\n'.format(baseName))
		elif line.startswith('OutputFile'):
			outCom.write('OutputFile\t{:s}_erase_ali.mrc\n'.format(baseName))
		elif line.startswith('ModelFile'):
			outCom.write('ModelFile\t{:s}_erase.fid\n'.format(baseName))			     
		else:
			outCom.write(line)
	outCom.close()
				     
def run_trimvol(baseName, tempCont):
	outCom = open('trimvol.com', 'w')
	outCom.write('$trimvol -f -rx {:s}_full_rec.mrc {:s}_rec.mrc'.format(baseName, baseName))
	outCom.close()
  	

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Batch tomo reconstruct using IMOD')
	parser.add_argument('--i', help='Input Tilt Series',required=True)
	parser.add_argument('--operation', help='Imod operation (newst/tilt/newst_3dfind/tilt_3dfind/findbeads3d/tilt_3dfind_reproject/golderaser/trimvol)',required=True)
	#parser.add_argument('--bin', help='Bin factor',required=True)			     
	parser.add_argument('--template', help='Template file from 1 reconstruction',required=True)
	parser.add_argument('--noproc', help='Number of processors',required=False, default=1)
	parser.add_argument('--excludeList', help='ExcludeList',required=False, default="")


	#parser.add_argument('--xsize', help='X size of image',required=True)	
	#parser.add_argument('--ysize', help='Y size of image',required=True)			     
	
	print('Only tested with 4.11 only')
	
	args = parser.parse_args()
	     	
	tempfile = open(args.template, 'r')
	tempCont = tempfile.readlines()
	operation = args.operation
	
	
	currentDir = os.getcwd()
	
	# Parallel
	# Init multiprocessing.Pool()
	#pool = mp.Pool(nocpu)

	tsList = glob.glob(args.i)
	for tiltseries in tsList:
		tsName = os.path.basename(tiltseries)
		if not tiltseries.endswith('.mrc'):
			print('---> Processing ' + tiltseries + ': ERROR! Only mrc file supported')
			break
		baseName = tsName.replace('.mrc','')
		tsPath = os.path.dirname(tiltseries)
		print('---> Processing ' + tiltseries + ' ...')
		print('Change dir to ' + tsPath)
		os.chdir(tsPath)
		if operation == 'eraser':
			run_eraser(baseName, tempCont)
			os.system('submfg eraser.com')
			shutil.move(baseName + '.mrc', baseName + '_orig.mrc')
			print('mv ' + baseName + '_fixed.mrc ' + baseName + '.mrc')
			shutil.move(baseName + '_fixed.mrc', baseName + '.mrc')
		elif operation == 'prenewst':
			run_prenewst(baseName, tempCont)
			os.system('submfg prenewst.com')
		elif operation == 'align':
			run_align(baseName, tempCont, args.excludeList)
			os.system('submfg align.com')
		elif operation == 'newst':
			run_newst(baseName, tempCont)
			os.system('submfg newst.com')
		elif operation == 'tilt':
			run_tilt(baseName, tempCont, args.excludeList)
			os.system('submfg tilt.com')
		elif operation == 'newst_3dfind':
			run_newst_3dfind(baseName, tempCont)
			os.system('submfg newst_3dfind.com')
		elif operation == 'tilt_3dfind':
			run_tilt_3dfind(baseName, tempCont, args.excludeList)
			os.system('submfg tilt_3dfind.com')
		elif operation == 'findbeads3d':
			run_findbeads3d(baseName, tempCont)
			os.system('submfg findbeads3d.com')
		elif operation == 'tilt_3dfind_reproject':
			run_tilt_3dfind_reproject(baseName, tempCont)
			os.system('submfg tilt_3dfind_reproject.com')
		elif operation == 'golderaser':
			run_golderaser(baseName, tempCont)
			os.system('submfg golderaser.com')
			shutil.move(baseName + '_ali.mrc', baseName + '_ali.mrc~')
			print('mv ' + baseName + '_erase_ali.mrc ' + baseName + '_ali.mrc')
			shutil.move(baseName + '_erase_ali.mrc', baseName + '_ali.mrc')
		elif operation == 'trimvol':
			run_trimvol(baseName, tempCont)
			os.system('submfg trimvol.com')
		else:
			print('Unknown option')
			exit(1)
		os.chdir(currentDir)
