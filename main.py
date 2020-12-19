import os
import glob
import csv
import numpy as np
from joblib import Parallel, delayed
from astropy.io import fits
from kep_main import keppca

from noise_fns.equal_mask import main as eq_mask_fn
from noise_fns.equal_mask_sig_clip_2 import main as eq_mask_sig_clip_2_fn
from noise_fns.equal_mask_sig_clip_3 import main as eq_mask_sig_clip_3_fn
from noise_fns.equal_pixel import main as eq_pixel_fn
from noise_fns.equal_pixel_sig_clip_2 import main as eq_pixel_sig_clip_2_fn
from noise_fns.equal_pixel_sig_clip_3 import main as eq_pixel_sig_clip_3_fn
from noise_fns.random_mask import main as rd_mask_fn
from noise_fns.random_mask_sig_clip_2 import main as rd_mask_sig_clip_2_fn
from noise_fns.random_mask_sig_clip_3 import main as rd_mask_sig_clip_3_fn


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(BASE_DIR, 'data')

INPUT_FOLDER_PATH = os.path.join(BASE_DIR, 'input_files')

OUTPUT_FOLDER = os.path.join(_DATA_PATH, 'outputs')
NORM_FILES_FOLDER = os.path.join(_DATA_PATH, 'norm_input_files')
NORM_REDUCED_FILES_FOLDER = os.path.join(_DATA_PATH, 'norm_reduced_input_files')
CSV_FOLDER = os.path.join(_DATA_PATH, 'csv')

NOISE_FNS = [
    eq_mask_fn,
    eq_mask_sig_clip_2_fn,
    eq_mask_sig_clip_3_fn,
    eq_pixel_fn,
    eq_pixel_sig_clip_2_fn,
    eq_pixel_sig_clip_3_fn,
    rd_mask_fn,
    rd_mask_sig_clip_2_fn,
    rd_mask_sig_clip_3_fn
]

SIGMA_LIST = [0.01, 0.02, 0.05, 0.08, 0.10, 0.25]

SAMPLE_SIZE = 1000

def make_req_dirs():
    req_folders = [
        _DATA_PATH, 
        OUTPUT_FOLDER, 
        NORM_FILES_FOLDER,
        NORM_REDUCED_FILES_FOLDER,
        CSV_FOLDER
    ]

    for folder in req_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

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
    
    for f in glob.glob(NORM_FILES_FOLDER + "/*.fits"):
        f_name = f.split('/')[-1].split('.')[0]
        pca_file_path = os.path.join(NORM_REDUCED_FILES_FOLDER, '%s_reduced.fits' % f_name) 
        if not os.path.exists(pca_file_path):
            keppca(f, outfile=pca_file_path, components='1-10', plotpca=False, nmaps=10)


def process_for_stats(lc, fn_folder, noisy_lc_folder):
        OUTPUT_HEADER = ['iter', 'mean', 'variance']

        out_file = open(os.path.join(CSV_FOLDER, lc, fn_folder, "%s.csv" % noisy_lc_folder), 'w')
        dict_writer = csv.DictWriter(out_file, OUTPUT_HEADER)
        dict_writer.writeheader()
        
        parent_lc = os.path.join(NORM_REDUCED_FILES_FOLDER, "%s_reduced.fits" % lc)
        path = os.path.join(OUTPUT_FOLDER, lc, fn_folder, "pca_files", noisy_lc_folder)

        stats_data = []

        for noisy_lc in glob.glob(path + "/*.fits"):
            hdul1 = fits.open(parent_lc)
            data1 = hdul1[1].data
            
            hdul2 = fits.open(noisy_lc)
            data2 = hdul2[1].data

            residual = np.divide(data2['PCA_FLUX'], data1['PCA_FLUX'])
    
            stats_data.append({
                'iter': noisy_lc.split('/')[-1].split('.')[0],
                'mean': np.nanmean(residual),
                'variance': np.nanvar(residual)
            })
        
        dict_writer.writerows(stats_data)
        out_file.flush()
        os.fsync(out_file.fileno())


def collect_stats():
    
    def fn_iterator():
        for lc in os.listdir(OUTPUT_FOLDER):
            for fn_folder in os.listdir(os.path.join(OUTPUT_FOLDER, lc)):
                out_path = os.path.join(CSV_FOLDER, lc, fn_folder)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                for noisy_lc_folder in os.listdir(os.path.join(OUTPUT_FOLDER, lc, fn_folder, "pca_files")):
                    yield lc, fn_folder, noisy_lc_folder 

    Parallel(n_jobs=-2, verbose=10, backend="multiprocessing")(
        delayed(process_for_stats)(lc, fn_folder, noisy_lc_folder) for lc, fn_folder, noisy_lc_folder in fn_iterator()
    )


def main():

    def fn_iterator():
        for f in glob.glob(NORM_FILES_FOLDER + "/*.fits"):
            print("Processing File: %s" % f)
            
            output_folder = os.path.join(OUTPUT_FOLDER, f.split('/')[-1].split('.')[0])
            for counter, noise_fn in enumerate(NOISE_FNS):
                
                print("*"*20, "[Noise Fn: %s OF %s]" % (counter + 1, len(NOISE_FNS)), "*"*20)
                
                for sig in SIGMA_LIST:
                    PCA_FILES_FOLDER = os.path.join(output_folder, noise_fn.__module__.split(".")[-1], 'pca_files', 'sigma_%s' % str(sig).replace('.', '_'))
                    RAW_FILES_FOLDER = os.path.join(output_folder, noise_fn.__module__.split(".")[-1], 'raw_files', 'sigma_%s' % str(sig).replace('.', '_'))

                    if not os.path.exists(PCA_FILES_FOLDER):
                        os.makedirs(PCA_FILES_FOLDER)
                    if not os.path.exists(RAW_FILES_FOLDER):
                        os.makedirs(RAW_FILES_FOLDER)

                    for iter_num in range(SAMPLE_SIZE): 
                        yield f, noise_fn, sig, iter_num, PCA_FILES_FOLDER, RAW_FILES_FOLDER

    Parallel(n_jobs=-2, verbose=10, backend="multiprocessing")(
        delayed(fn)(f, sig, iter_num, PCA_FILES_FOLDER, RAW_FILES_FOLDER) for f, fn, sig, iter_num, PCA_FILES_FOLDER, RAW_FILES_FOLDER in fn_iterator()
    )
   

if __name__ == '__main__':
    make_req_dirs()
    preprocess()
    main()
    collect_stats()
    print('\n\n', "-" * 100, '\n', 'Run Complete\n', "-" * 100, '\n', '=== STATS ===')

