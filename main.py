import os
import glob
import csv
import numpy as np
from joblib import Parallel, delayed
from astropy.io import fits

from noise_fns.equal_mask import main as eq_mask_fn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(BASE_DIR, 'data')

INPUT_FOLDER_PATH = os.path.join(BASE_DIR, 'input_files')

OUTPUT_FOLDER = os.path.join(_DATA_PATH, 'outputs')
NORM_FILES_FOLDER = os.path.join(_DATA_PATH, 'norm_input_files')

NOISE_FNS = [
    eq_mask_fn,
]

SIGMA_LIST = [0.01, 0.02, 0.05, 0.08, 0.10, 0.25]

SAMPLE_SIZE = 10

def make_req_dirs():
    req_folders = [
        _DATA_PATH, 
        OUTPUT_FOLDER, 
        NORM_FILES_FOLDER, 
    ]

    for folder in req_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)

def preprocess():
    for f in glob.glob(INPUT_FOLDER_PATH + "/*.fits"):
        hdul1 = fits.open(f)
        data1 = hdul1[1].data
        median = np.nanmedian(data1['FLUX'])
        data1['FLUX'] = np.divide(data1['FLUX'], median)
        f_name = f.split('/')[-1].split('.')[0]
        try:
            hdul1.writeto(os.path.join(NORM_FILES_FOLDER, '%s_norm.fits' % f_name))
        except OSError:
            pass


def collect_stats():
    ...

def main():
    make_req_dirs()
    preprocess()

    def fn_iterator():
        for f in glob.glob(NORM_FILES_FOLDER + "/*.fits"):
            output_folder = os.path.join(OUTPUT_FOLDER, f.split('/')[-1].split('.')[0])
            for noise_fn in NOISE_FNS:
                for sig in SIGMA_LIST:
                    for i in range(SAMPLE_SIZE): 
                        yield f, noise_fn, sig, i, output_folder

    Parallel(n_jobs=-2, verbose=10, backend="multiprocessing")(
        delayed(fn)(f, sig, i, output_folder) for f, fn, sig, i, output_folder in fn_iterator()
    )

    collect_stats()

    print('\n\n', "-" * 100, '\n', 'Run Complete\n', "-" * 100, '\n', '=== STATS ===')
   

if __name__ == '__main__':
    main()
