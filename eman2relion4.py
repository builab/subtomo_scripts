#/usr/bin/python3
# Convert a list of EMAN2 3D coordinate to relion4
# HB 12/2021

import numpy as np
import pandas as pd
import argparse, os, re

from eulerangles import euler2euler
from eulerangles import convert_eulers

def write_star_4(dfin, outfile, tableType):
	out = open(outfile, 'w')
	out.write("# version 30001 from aa\n\n")
	out.write("data_" + tableType +"\n\n")
	out.write("loop_\n")
	for i in range(len(dfin.columns)):
		out.write('_rln{:s} #{:d}\n'.format(dfin.columns[i], i+1))
	out.write(dfin.to_string(index=False, header=False))
	out.write("\n")
	out.close()

def preprocess_spider_doc(spiderdoc):
	cmd = 'sed -i \'/^ ;/d\' ' + spiderdoc
	os.system(cmd)
	
def preprocess_bstar(starFile):
	cmd = 'grep \'^\\s*[0-9]\' ' + starFile + ' > ' + starFile.replace('.star', '.txt')
	os.system(cmd)


"""Convert aa doc & star to dynamo table"""
def aa_to_relion(starFile, docFile, tomoName, tomoNo, binFactor, pixelSize, doubletId):
	# Read the doc file
	header_list=["no", "norec", "phi", "theta", "psi", "OriginX", "OriginY", "OriginZ", "cc"]
	df = pd.read_csv(docFile, delim_whitespace=True, names=header_list)
	fulldata = df.to_numpy()

	# Extract phi, theta, psi (AA format) and reverse sign of phi & psi
	eulers_zyz = fulldata[:, 2:5]*-1
	eulers_zyz[:,1] = eulers_zyz[:,1]*-1

	eulers_dynamo = euler2euler(eulers_zyz, source_axes='zyz', source_intrinsic=True, source_right_handed_rotation=True,
								target_axes='zxz', target_intrinsic=True,target_right_handed_rotation=True,invert_matrix=False)

	# Read the star file (ignore header for now)
	star_header = ["no", "c2", "c3", "c4", "CoordinateX", "CoordinateY", "CoordinateZ", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16"]
	df2 = pd.read_csv(starFile, delim_whitespace=True, names=star_header)
	fullstar = df2.to_numpy()

	# Extract origin
	origin = fullstar[:, 4:7]
	nrows, ncols = origin.shape

	# Hard Code Here
	header_list = ["TomoName", "TomoParticleId", "TomoManifoldIndex", "CoordinateX", "CoordinateY", "CoordinateZ", "OriginXAngst", "OriginYAngst", "OriginZAngst", "AngleRot", "AngleTilt", "AnglePsi", "ClassNumber", "RandomSubset"]
	df_relion = pd.DataFrame(columns = header_list)
	df_relion['TomoParticleId'] = np.arange(len(df2), dtype=np.int16) + 1
	df_relion['TomoManifoldIndex'] = np.ones(len(df2['CoordinateX']), dtype=np.int16)*doubletId	
	df_relion['CoordinateX'] = df2['CoordinateX']*binFactor;
	df_relion['CoordinateY'] = df2['CoordinateY']*binFactor;
	df_relion['CoordinateZ'] = df2['CoordinateZ']*binFactor;
	# To adjust originXYZ
	df_relion['OriginXAngst'] = np.zeros(len(df_relion['CoordinateX']))
	df_relion['OriginYAngst'] = np.zeros(len(df_relion['CoordinateX']))
	df_relion['OriginZAngst'] = np.zeros(len(df_relion['CoordinateX']))

	# Reset angle for debug
	eulers_relion = convert_eulers(eulers_dynamo, source_meta='dynamo', target_meta='warp')
	df_relion['AngleRot'] = eulers_relion[:,0]
	df_relion['AngleTilt'] = eulers_relion[:,1]
	df_relion['AnglePsi'] = eulers_relion[:,2]


	df_relion['ClassNumber'] = np.ones(len(df_relion['CoordinateX']), dtype=np.int8)

	for i in range(len(df2['CoordinateX'])):
		df_relion.loc[i, ('TomoName')] = tomoName

	a = np.empty((len(df_relion['CoordinateX']),), dtype=np.int8)
	a[::2] = 1
	a[1::2] = 2

	df_relion['RandomSubset'] = a
	return df_relion


if __name__=='__main__':
	# get name of input starfile, output starfile, output stack file
	print('Script to convert from AxonemeAlign to Relion. HB 2021')
	
	parser = argparse.ArgumentParser(description='Convert doc & star file to Relion 4.0 input file')
	parser.add_argument('--i', help='Input folder of eman3D coordinate',required=True)
	parser.add_argument('--ostar', help='Output star file',required=True)
	parser.add_argument('--angpix', help='Input pixel size',required=True)
	parser.add_argument('--bin', help='Bin of current coordinate',required=True)
	parser.add_argument('--frac_dose', help='Tomo fractional dose',required=True, default=2)


	args = parser.parse_args()
	listDoublet = open(args.i, 'r')
	pixelSize = float(args.angpix)
	binFactor = float(args.bin)
		
	tomoList = {}
	tomoNo = 0;
	df_all = None
	
	# Template for tomo_description
	orderList = 'input/order_list.csv'
	
	tomo_header_list = ["TomoName", "TomoTiltSeriesName", "TomoImportCtfFindFile", "TomoImportImodDir", "TomoImportFractionalDose", "TomoImportOrderList", "TomoImportCulledFile"]
	df_tomo = pd.DataFrame(columns = tomo_header_list)
		
	for line in listDoublet:   
		if line.startswith('#'):
			continue
		record = line.split()
		# Check tomo
		# This is not so robust for tomoa & tomob name yet
		tomoSubName = record[0].replace('_ida_v1', '')
		tomoSubName = tomoSubName[:-4]
		# Replace a, b, c in case. Not exact more than 3 tomo
		tomoName = re.sub('[a-z]$', '', tomoSubName)
	
		doubletId = int(record[1][-1])

		if tomoList.get(tomoName) == None:
			print(tomoName)
			tomoNo += 1
			tomoList[tomoName] = tomoNo
			df_tomo.loc[tomoNo-1, 'TomoName'] = tomoName
			df_tomo.loc[tomoNo-1, 'TomoTiltSeriesName'] = 'tomograms/' + tomoName + '/' + tomoName + '.mrc'
			df_tomo.loc[tomoNo-1, 'TomoImportCtfFindFile'] = 'tomograms/' + tomoName + '/' + tomoName + '_output.txt'
			df_tomo.loc[tomoNo-1, 'TomoImportImodDir'] = 'tomograms/' + tomoName
			df_tomo.loc[tomoNo-1, 'TomoImportFractionalDose'] = args.frac_dose
			df_tomo.loc[tomoNo-1, 'TomoImportOrderList'] = orderList
			df_tomo.loc[tomoNo-1, 'TomoImportCulledFile'] = 'tomograms/' + tomoName + '/' + tomoName + '_culled.mrc'
			
			
		print('   -->' + str(doubletId))
		# This part need to be fixed
		emanFile = 'star/' + record[1]  + '.star'
		# Convert
		df_relion = eman3d_to_relion(emanFile, tomoName, tomoNo, binFactor, pixelSize)

		if df_all is None:
			df_all = df_relion.copy()
		else:
			df_all = df_all.append(df_relion)

	# Renumber
	df_all['TomoParticleId'] = np.arange(len(df_all), dtype=np.int16) + 1
	print("Writing " + args.ostar)
	write_star_4(df_all, args.ostar, 'particles') 
	print("Writing tomograms_descr.star")
	write_star_4(df_tomo, 'tomograms_descr.star', '')
