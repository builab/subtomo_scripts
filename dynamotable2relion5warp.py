#/usr/bin/python3
# Convert dynamo table to relion 5
# A lot of the code is from dynamo2m
# Now using dynamotable & starfile from Alisterburt
# Conversion based on the tomograms.doc, therefore, it will be a lot easier to organize than dynamo2relion from Pyle
# https://pypi.org/project/dynamo2relion/
#
# Huy Bui, McGill 2025, not yet tested

import numpy as np
import pandas as pd
import argparse, os
import starfile
import dynamotable
import re

from eulerangles import convert_eulers

def dynamo2relion5warp (table, output_star_file, binFactor, imageSize, helicalCol, tomostarDir):
	# Loop through every tomogram and then write particles
	tableLookup = ["tag", "aligned", "averaged", "dx", "dy", "dz", "tdrot", "tilt", "narot", "cc", "cc2", "cpu", "ftype", "ymintilt", "ymaxtilt", "xmintilt", "xmaxtilt", "fs1", "fs2", "tomo", "reg", "class", "annotation", "x", "y", "z", "dshift", "daxis", "dnarot", "dcc", "otag", "npar", "ref", "sref", "apix", "def", "eig1", "eig2"]
	# Prep data for star file
	df_particles = None
	
	
	# Make sorted table
	sorted_table = table.sort_values(by='tomo').drop_duplicates(subset=['tomo'])

	#Loop through table
	for tomoNo in sorted_table['tomo']:
		# Select all particles belong to the same tomo
		tomoData = table[table['tomo'] == tomoNo]
		tomoName = sanitise_warp_tomo_name(tomoData.iloc[0]['tomo_file']).replace('.tomostar', '')
		print('Reading data from tomo {:s}'.format(tomoName))
		#print(tomoData['tomo_file'])
		
		if helicalCol > 0:
			header_list = ["rlnTomoName", "rlnTomoParticleId", "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnTomoParticleName", "rlnOpticsGroup", "rlnImageName", "rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst", "rlnTomoVisibleFrames", "rlnClassNumber", "rlnHelicalTubeID", "rlnRandomSubset"]
		else:
			header_list = ["rlnTomoName", "rlnTomoParticleId", "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnTomoParticleName", "rlnOpticsGroup", "rlnImageName", "rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst", "rlnTomoVisibleFrames", "rlnClassNumber"]
		
		df = pd.DataFrame(columns = header_list)
		
		# extract xyz into dict with relion style headings
		for axis in ('x', 'y', 'z'):
			heading = f'rlnCoordinate{axis.upper()}'
			shift_axis = f'd{axis}'
			df[heading] = tomoData[axis] + tomoData[shift_axis] # Relion5 Warp doesn't use absolute coordinate
		
		# Need to reset for every tomogram
		#print(np.arange(len(df['rlnCoordinateX']), dtype=np.int16))
		df['rlnTomoParticleId'] = np.arange(len(df['rlnCoordinateX']), dtype=np.int16) + 1
			
		#print(df['rlnTomoParticleId'])

		# extract and convert eulerangles
		eulers_dynamo = tomoData[['tdrot', 'tilt', 'narot']].to_numpy()
		eulers_warp = convert_eulers(eulers_dynamo, source_meta='dynamo',target_meta='warp')
		# Ensure eulers_relion is always 2-dimensional
		if eulers_warp.ndim == 1:
			eulers_warp = eulers_warp.reshape(1, -1)
			
		df['rlnAngleRot'] = eulers_warp[:, 0]
		df['rlnAngleTilt'] = eulers_warp[:, 1]
		df['rlnAnglePsi'] = eulers_warp[:, 2]
	
		df['rlnClassNumber'] = np.ones(len(df['rlnCoordinateX']), dtype=np.int16)
		
		# To adjust originXYZ
		df['rlnOriginXAngst'] = np.zeros(len(df['rlnCoordinateX']))
		df['rlnOriginYAngst'] = np.zeros(len(df['rlnCoordinateX']))
		df['rlnOriginZAngst'] = np.zeros(len(df['rlnCoordinateX']))
		
		df['rlnOpticsGroup'] = np.zeros(len(df['rlnCoordinateX']), dtype=np.int16) + int(tomoNo)
		#print(df['rlnOpticsGroup'])
	
		if helicalCol > 0:
			df['rlnHelicalTubeID'] = tomoData[tableLookup[helicalCol - 1]]
			#print(df['rlnHelicalTubeID'])
			# Temporary Fix random subset base on tomogram, Need to make it based on HelicaTubeID later
			assignedSet = 1
			randomSubset = tomoData['annotation'].copy()
			for helicalTubeID in randomSubset.unique():
				randomSubset[randomSubset == helicalTubeID] = assignedSet;
				if assignedSet == 1:
					assignedSet = 2
				else:
					assignedSet = 1
				
			#print(randomSubset)
			df['rlnRandomSubset'] = randomSubset
			
		df['rlnTomoName'] = tomoData['tomo_file'].apply(sanitise_warp_tomo_name)
		#print(df['rlnTomoName'])
			
		# Look up how many tilt is used
		df_tomostar = starfile.read(tomostarDir + '/' + df.iloc[0]['rlnTomoName'])
		visible_frames = f"[{','.join(['1'] * len(df_tomostar))}]"
		
		df['rlnTomoParticleName'] = tomoName + '/' + df['rlnTomoParticleId'].apply(lambda x: f"{x}")
		df['rlnImageName'] = '../warp_tiltseries/particleseries/' + tomoName + '/' + tomoName + f"_{pixelSize*binFactor:02}" + "A_" +  df['rlnTomoParticleId'].apply(lambda x: f"{x:06}") + ".mrcs"
		df['rlnTomoVisibleFrames'] = visible_frames  # Replace with your desired number		

		#for i in df.index:
		#	#print(i)
		#	df.loc[i, ('rlnTomoParticleName')] = tomoName + '/' + str(int(df.iloc[i]['rlnTomoParticleId']))
		#	df.loc[i, ('rlnImageName')] = '../warp_tiltseries/particleseries/' + tomoName + '/' + tomoName + f"_{pixelSize*binFactor:02}" + "A_" +  f"{int(df.iloc[i]['rlnTomoParticleId']):06}" + ".mrcs"
		#	#print(df.loc[i, ('rlnImageName')])
		#	df.loc[i, ('rlnTomoVisibleFrames')] = visible_frames  # Replace with your desired number		
		
		# Fix printing problem
		#df['rlnTomoParticleId'] = df['rlnTomoParticleId'].astype(int).astype(str)

		if df_particles is None:
			df_particles = df.copy()
		else:
			df_particles = pd.concat([df_particles, df], ignore_index=True)

		print(df)
		#print(len(df_particles))
	
		#df_particles['rlnTomoParticleId'] = df_particles.groupby('rlnTomoName').cumcount() + 1
		#for index, value in df_particles['rlnTomoParticleId'].items():
	 	#	try:
	 	#		int_value = int(value)
	 	#	except ValueError:
	 	#		print(f"Error converting value at index {index}: {value}")
				
		#df_particles['rlnTomoParticleId'] = df_particles['rlnTomoParticleId'].astype(int)
		#df_particles['rlnTomoParticleName'] = df_particles['rlnTomoName'].str.replace('.tomostar', '', regex=False) + '/' + df_particles['rlnTomoParticleId'].astype(str)
	
	#df_particles['rlnTomoParticleId'] = df_particles['rlnTomoParticleId'].astype(int).astype(str)
	#print(df_particles['rlnTomoParticleId'])

	return df_particles 
	
def dynamo2tomo(table, pixelSize, binFactor, imageSize):
	# Read table file into the opticsGroup part
	tableLookup = ["tag", "aligned", "averaged", "dx", "dy", "dz", "tdrot", "tilt", "narot", "cc", "cc2", "cpu", "ftype", "ymintilt", "ymaxtilt", "xmintilt", "xmaxtilt", "fs1", "fs2", "tomo", "reg", "class", "annotation", "x", "y", "z", "dshift", "daxis", "dnarot", "dcc", "otag", "npar", "ref", "sref", "apix", "def", "eig1", "eig2"]

	tomo_header_list = ["rlnOpticsGroup", "rlnOpticsGroupName", "rlnSphericalAberration", "rlnVoltage", "rlnTomoTiltSeriesPixelSize", "rlnCtfDataAreCtfPremultiplied", "rlnImageDimensionality", "rlnTomoSubtomogramBinning", "rlnImagePixelSize", "rlnImageSize", "rlnAmplitudeContrast"]
	df_tomo = pd.DataFrame(columns = tomo_header_list)
	
	sorted_table = table.sort_values(by='tomo').drop_duplicates(subset=['tomo'])
	for tomoNo in sorted_table['tomo']:
			df_tomo.loc[tomoNo-1, 'rlnOpticsGroup'] = tomoNo
			df_tomo.loc[tomoNo-1, 'rlnOpticsGroupName'] = 'opticsGroup' + str(tomoNo)
			df_tomo.loc[tomoNo-1, 'rlnSphericalAberration'] = 2.7
			df_tomo.loc[tomoNo-1, 'rlnVoltage'] = 300
			df_tomo.loc[tomoNo-1, 'rlnTomoTiltSeriesPixelSize'] = pixelSize
			df_tomo.loc[tomoNo-1, 'rlnCtfDataAreCtfPremultiplied'] = 1
			df_tomo.loc[tomoNo-1, 'rlnImageDimensionality'] = 2
			df_tomo.loc[tomoNo-1, 'rlnTomoSubtomogramBinning'] = binFactor
			df_tomo.loc[tomoNo-1, 'rlnImagePixelSize'] = pixelSize*binFactor
			df_tomo.loc[tomoNo-1, 'rlnImageSize'] = imageSize;
			df_tomo.loc[tomoNo-1, 'rlnAmplitudeContrast'] = 0.07
	
	return df_tomo
	
def sanitise_warp_tomo_name(micrograph_name: str) -> str:
	"""
	Replaces tomogram name from Warp reconstructions with corresponding name file if appropriate
	"""
	micro = re.sub(r"_\d\.\d+Apx.mrc", ".tomostar", micrograph_name)
	return re.sub(r"^.*\/", "", micro)
	

if __name__=='__main__':
   	# get name of input starfile, output starfile, output stack file
	print('Script to convert from Dynamo to Relion 4')
	
	parser = argparse.ArgumentParser(description='Convert tbl file to Relion 4.0 input file')
	parser.add_argument('--tbl', help='Input table file',required=True)
	parser.add_argument('--tomodoc', help='Input tomo doc file',required=True)
	parser.add_argument('--ostar', help='Output star file',required=True)
	parser.add_argument('--angpix', help='Original pixel size',required=True)
	parser.add_argument('--bin', help='Bin of current tomo',required=True)
	parser.add_argument('--imagesize', help='Subtomo image size',required=True, default=2)
	parser.add_argument('--path_tomostar', help='Path to tomostar',required=True)
	parser.add_argument('--helicalCol', help='Column from table to used as helicalID',required=False,default=0)


	args = parser.parse_args()
	pixelSize = float(args.angpix)
	imageSize = float(args.imagesize)
	binFactor = float(args.bin)
	tomostarDir = args.path_tomostar
	tomoDoc = open(args.tomodoc, 'r')
	helicalCol = int(args.helicalCol)



	df_all = None
	
	# Generate the opticsGroup part
	table = dynamotable.read(args.tbl, args.tomodoc)
	df_tomo = dynamo2tomo(table, pixelSize, binFactor, imageSize)
	
	# Convert Coordinate
	df_particles = dynamo2relion5warp(table, args.ostar, binFactor, imageSize, int(args.helicalCol), tomostarDir)

		
	general_df = {};
	general_df['rlnTomoSubTomosAre2DStacks'] = 1
	
	print("Writing " + args.ostar)
	starfile.write({'general': general_df, 'optics': df_tomo, 'particles': df_particles}, args.ostar)

