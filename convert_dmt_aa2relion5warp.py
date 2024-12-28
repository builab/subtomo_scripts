#/usr/bin/python3
# Convert the entire cilia from AA to relion4
# Must run adjustOrigin from AA before
# Add HelicalTubeID now
# Making compatible with both Macs & Linux for sed command
# Making compatible with new python 3.9
# Make compatible with eulers_relion with one row only
# Read TomoVisibleFrames from tomostar file
# HB 08/2022


import numpy as np
import pandas as pd
import argparse, os, re
import starfile

from eulerangles import euler2euler
from eulerangles import convert_eulers

def preprocess_spider_doc(spiderdoc):
	cmd = 'sed -i \'\' \'/^ ;/d\' ' + spiderdoc
	os.system(cmd)
	
def preprocess_bstar(starFile):
	cmd = 'grep \'^\\s*[0-9]\' ' + starFile + ' > ' + starFile.replace('.star', '.txt')
	os.system(cmd)

"""Convert aa doc & star to dynamo table"""
def aa_to_relion5warp(starFile, docFile, tomoName, tomoNo, binFactor, pixelSize, doubletId):
	# Read the doc file
	# Question, do we need TomoVisibleFrames
	header_list=["no", "norec", "phi", "theta", "psi", "OriginX", "OriginY", "OriginZ", "cc"]
	df = pd.read_csv(docFile, sep='\s+', names=header_list)
	fulldata = df.to_numpy()

	# Extract phi, theta, psi (AA format) and reverse sign of phi & psi
	eulers_zyz = fulldata[:, 2:5]*-1
	eulers_zyz[:,1] = eulers_zyz[:,1]*-1

	eulers_dynamo = euler2euler(eulers_zyz, source_axes='zyz', source_intrinsic=True, source_right_handed_rotation=True,
								target_axes='zxz', target_intrinsic=True,target_right_handed_rotation=True,invert_matrix=False)

	# Read the star file (ignore header for now)
	star_header = ["no", "c2", "c3", "c4", "CoordinateX", "CoordinateY", "CoordinateZ", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16"]
	df2 = pd.read_csv(starFile, sep='\s+', names=star_header)
	fullstar = df2.to_numpy()

	# Extract origin
	origin = fullstar[:, 4:7]
	nrows, ncols = origin.shape

	# Hard Code Here
	header_list = ["TomoName", "TomoParticleId", "CoordinateX", "CoordinateY", "CoordinateZ", "AngleRot", "AngleTilt", "AnglePsi", "TomoParticleName", "OpticsGroup", "ImageName", "OriginXAngst", "OriginYAngst", "OriginZAngst", "TomoVisibleFrames", "ClassNumber", "HelicalTubeID", "RandomSubset"]
	df_relion = pd.DataFrame(columns = header_list)
	df_relion['TomoParticleId'] = np.arange(len(df2), dtype=np.int16) + 1
	df_relion['HelicalTubeID'] = np.ones(len(df2['CoordinateX']), dtype=np.int16)*doubletId	
	df_relion['CoordinateX'] = df2['CoordinateX'];
	df_relion['CoordinateY'] = df2['CoordinateY'];
	df_relion['CoordinateZ'] = df2['CoordinateZ'];
	
	# To adjust originXYZ
	df_relion['OriginXAngst'] = np.zeros(len(df_relion['CoordinateX']))
	df_relion['OriginYAngst'] = np.zeros(len(df_relion['CoordinateX']))
	df_relion['OriginZAngst'] = np.zeros(len(df_relion['CoordinateX']))
	
	df_relion['OpticsGroup'] = np.zeros(len(df_relion['CoordinateX'])) + tomoNo
	
	# Reset angle for debug
	eulers_relion = convert_eulers(eulers_dynamo, source_meta='dynamo', target_meta='warp')
	# Ensure eulers_relion is always 2-dimensional
	if eulers_relion.ndim == 1:
		eulers_relion = eulers_relion.reshape(1, -1)
		
	df_relion['AngleRot'] = eulers_relion[:,0]
	df_relion['AngleTilt'] = eulers_relion[:,1]
	df_relion['AnglePsi'] = eulers_relion[:,2]


	df_relion['ClassNumber'] = np.ones(len(df_relion['CoordinateX']), dtype=np.int8)

	# Look up how many tilt is used
	df_tomostar = starfile.read('tomostar/' + tomoName + '.tomostar' )
	visible_frames = f"[{','.join(['1'] * len(df_tomostar))}]"

	for i in range(len(df2['CoordinateX'])):
		df_relion.loc[i, ('TomoName')] = tomoName + '.tomostar'
		df_relion.loc[i, ('TomoParticleName')] = tomoName + '/' + str(df_relion.loc[i, ('TomoParticleId')])
		df_relion.loc[i, ('ImageName')] = '../warp_tiltseries/particleseries/' + tomoName + '/' + tomoName + f"_{pixelSize*binFactor:02}" + "A_" +  f"{df_relion.loc[i, ('TomoParticleId')]:06}" + ".mrcs"
		df_relion.loc[i, ('TomoVisibleFrames')] = visible_frames  # Replace with your desired number

	a = np.empty((len(df_relion['CoordinateX']),), dtype=np.int8)
	a[::2] = 1
	a[1::2] = 2

	df_relion['RandomSubset'] = a
	return df_relion


if __name__=='__main__':
	# get name of input starfile, output starfile, output stack file
	print('Script to convert from AxonemeAlign to Relion5 Warp. HB 2024')
	print('All the tomostars must be copy in tomostar/')
	parser = argparse.ArgumentParser(description='Convert doc & star file to Relion 4.0 input file')
	parser.add_argument('--i', help='Input list file',required=True)
	parser.add_argument('--ostar', help='Output star file',required=True)
	parser.add_argument('--angpix', help='Input pixel size',required=True)
	parser.add_argument('--imagesize', help='Input pixel size',required=True)
	parser.add_argument('--bin', help='Bin of current tomo',required=True)

	args = parser.parse_args()
	listDoublet = open(args.i, 'r')
	pixelSize = float(args.angpix)
	imageSize = float(args.imagesize)
	binFactor = float(args.bin)
		
	tomoList = {}
	tomoNo = 0;
	df_all = None
	
	# Template for tomo_description
	orderList = 'input/order_list.csv'
	
	tomo_header_list = ["OpticsGroup", "OpticsGroupName", "SphericalAberration", "Voltage", "TomoTiltSeriesPixelSize", "CtfDataAreCtfPremultiplied", "ImageDimensionality", "TomoSubtomogramBinning", "ImagePixelSize", "ImageSize", "AmplitudeContrast"]
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
			df_tomo.loc[tomoNo-1, 'OpticsGroup'] = tomoNo
			df_tomo.loc[tomoNo-1, 'OpticsGroupName'] = 'opticsGroup' + str(tomoNo)
			df_tomo.loc[tomoNo-1, 'SphericalAberration'] = 2.7
			df_tomo.loc[tomoNo-1, 'Voltage'] = 300
			df_tomo.loc[tomoNo-1, 'TomoTiltSeriesPixelSize'] = pixelSize
			df_tomo.loc[tomoNo-1, 'CtfDataAreCtfPremultiplied'] = 1
			df_tomo.loc[tomoNo-1, 'ImageDimensionality'] = 2
			df_tomo.loc[tomoNo-1, 'TomoSubtomogramBinning'] = binFactor
			df_tomo.loc[tomoNo-1, 'ImagePixelSize'] = pixelSize*binFactor
			df_tomo.loc[tomoNo-1, 'ImageSize'] = imageSize;
			df_tomo.loc[tomoNo-1, 'AmplitudeContrast'] = 0.07

			
		print('   -->' + str(doubletId))
		# This part need to be fixed
		starFile = 'star_corr/' + record[1]  + '.star'
		docFile = 'doc_corr/doc_total_' + record[0] + '.spi'
		# Remove the comment in spider file
		preprocess_bstar(starFile)
		preprocess_spider_doc(docFile)
		# Convert
		df_relion = aa_to_relion5warp(starFile.replace('.star', '.txt'), docFile, tomoName, tomoNo, binFactor, pixelSize, doubletId)

		if df_all is None:
			df_all = df_relion.copy()
		else:
			#df_all = df_all.append(df_relion)
			df_all = pd.concat([df_all, df_relion], ignore_index=True)


	general_df = {};
	general_df['TomoSubTomosAre2DStacks'] = 1
	particles_df = {}
	
	particles_df = df_all
	
	# Renumber
	df_all['TomoParticleId'] = np.arange(len(df_all), dtype=np.int16) + 1
	print("Writing " + args.ostar)
	starfile.write({'general': general_df, 'optics': df_tomo, 'particles': particles_df}, args.ostar)
