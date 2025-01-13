#!/usr/bin/env python3
# NOT TESTED YET

import os
from multiprocessing import Pool
import mrcfile
import torch
from fidder.predict import predict_fiducial_mask
from fidder.erase import erase_masked_region

#### User Defined Parameters ####
parent_dir = '/storage/builab/Thibault/20240905_SPEF1_MT_TS/warp_frameseries/'
angpix = 2.12
thresh = 0.5


def check_directories(parent_dir):
    for subdir in subdirs:
        dir_path = os.path.join(parent_dir, subdir)
        if os.path.exists(dir_path):
            print(dir_path + ' found.')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(dir_path + ' created.')

    os.makedirs(os.path.join(parent_dir, 'average_erased'), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, 'average', 'even_erased'), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, 'average', 'odd_erased'), exist_ok=True)

def make_mask(filename, parent_dir):
    """ Apply fidder's predict_fidcucial_mask function to a single micrograph and save the resultant mask as a new mrc file.

    Args:
        filename (str) : The name of the micrograph to process.
        parent_dir (str) : The directory containing micrographs in /average and frame-split halves in 
            /average/even, and /average/odd subdirectories.
    """
    mic_path = os.path.join(parent_dir, 'average', filename)
    mask_path = os.path.join(parent_dir, 'average', 'mask', filename)

    image = torch.tensor(mrcfile.read(mic_path))

    mask, probabilities = predict_fiducial_mask(
        image, pixel_spacing=angpix, probability_threshold=thresh
    )

    mask_uint8 = mask.to(torch.uint8)

    os.makedirs(os.path.dirname(mask_path), exist_ok=True)

    with mrcfile.new(mask_path, overwrite=True) as mrc:
        mrc.set_data(mask_uint8.numpy())

    print(mask_path + '/' + filename + ' mask made.')

def erase_gold(filename, parent_dir, subdir):
    """Apply fidder's erase_masked_region function to a single frame and save the result as a new mrc file.
        
    Args:
        filename (str): The name of the micrograph to process.
        parent_dir (str) : The directory containing micrographs in /average and frame-split halves in
            /average/even, and /average/odd subdirectories.
    """
    mask_path = os.path.join(parent_dir, 'average', 'mask', filename)
    
    mic_path  = os.path.join(parent_dir, subdir, filename)

    if subdir == 'average':
        output_subdir = os.path.join(parent_dir, 'average_erased')
    else:
        output_subdir = os.path.join(parent_dir, subdir + '_erased')

    os.makedirs(output_subdir, exist_ok=True)

    image = torch.tensor(mrcfile.read(mic_path))
    mask = torch.tensor(mrcfile.read(mask_path))

    erased_image = erase_masked_region(image=image, mask=mask)

    mrc_output_path = os.path.join(output_subdir, filename)
    with mrcfile.new(mrc_output_path, overwrite=True) as mrc:
        mrc.set_data(erased_image.numpy())

    print(output_subdir +  '/' + filename + ' completed')

def process_gold(subdir, parent_dir):
    filenames = [i for i in os.listdir(os.path.join(parent_dir, subdir)) if i.endswith('.mrc')]
    with Pool(48) as p:
        p.starmap(erase_gold, [(filename, parent_dir, subdir) for filename in filenames])
    print('################################ all gold erased for ' + subdir + ' ################################')


subdirs = ['average', 'average/even', 'average/odd']

if __name__ == "__main__":
    check_directories(parent_dir)

    filenames = [i for i in os.listdir(os.path.join(parent_dir, 'average')) if i.endswith('.mrc')]
    for filename in filenames:
        mask_path = os.path.join(parent_dir, 'average','mask', filename)
        if not os.path.exists(mask_path):
            make_mask(filename, parent_dir)
        else:
            print(f'{filename} aleady exists')
    print('################################ mask processing complete ################################')

    for subdir in subdirs:
        process_gold(subdir, parent_dir)
