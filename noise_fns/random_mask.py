import os
from astropy.io import fits
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from kep_main import keppca


def main(src_file, sigma, iter_num, PCA_FILES_FOLDER, RAW_FILES_FOLDER):
    print("Processing ........... Sigma: %s, Iter: %s" % (sigma, iter_num))
    
    hdul1 = fits.open(src_file)
    data1 = hdul1[1].data

    shape = data1['FLUX'].shape
    size = data1['FLUX'].size
    
    mu = 0
    s = np.random.normal(mu, sigma, size)
    s_reshape = np.reshape(s, shape)
    
    data1['FLUX'] = np.add(s_reshape, data1['FLUX'])
    
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

