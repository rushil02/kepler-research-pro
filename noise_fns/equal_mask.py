import os
from astropy.io import fits
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from kep_main import keppca


def main(src_file, sigma, iter_num, output_folder):
    print("Processing [%s] ........... %s, %s" % (src_file, sigma, iter_num))
    PCA_FILES_FOLDER = os.path.join(output_folder, 'equal_mask', 'pca_files', 'sigma_%s' % str(sigma).replace('.', '_'))
    RAW_FILES_FOLDER = os.path.join(output_folder, 'equal_mask', 'raw_files', 'sigma_%s' % str(sigma).replace('.', '_'))
    
    if not os.path.exists(PCA_FILES_FOLDER):
        os.makedirs(PCA_FILES_FOLDER)
    if not os.path.exists(RAW_FILES_FOLDER):
        os.makedirs(RAW_FILES_FOLDER)
    
    hdul1 = fits.open(src_file)
    data1 = hdul1[1].data

    shape = data1['FLUX'].shape

    mu = 0
    s = np.random.normal(mu, sigma, 306)
    s_master_clip = np.zeros(shape)

   
    for i, frame in enumerate(s_master_clip):
        for j, x in enumerate(frame):
            for k, y in enumerate(x):
                s_master_clip[i][j][k] = s[i]

    data1['FLUX'] = np.add(s_master_clip, data1['FLUX'])
    
    raw_file_path = os.path.join(RAW_FILES_FOLDER, '%04d.fits' % iter_num)
    pca_file_path = os.path.join(PCA_FILES_FOLDER, '%04d.fits' % iter_num) 

    if not os.path.exists(pca_file_path):
        try:
            hdul1.writeto(raw_file_path)
        except OSError:
            os.remove(raw_file_path)
            hdul1.writeto(raw_file_path)
        finally:
            keppca(raw_file_path, outfile=pca_file_path, components='1-10', plotpca=False, nmaps=10)

