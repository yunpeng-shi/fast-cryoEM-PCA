"""
Demo Code for Fast PCA on experimental data (EMPIAR-10081)
"""

import logging
from aspire.source.relion import RelionSource
import numpy as np
from fle_2d_single import FLEBasis2D
from fast_pca import FastPCA

logger = logging.getLogger(__name__)
# Set input path and files and initialize other parameters
DATA_FOLDER = './data'
STARFILE_IN = './data/particle_stacks/data.star'

MAX_ROWS = None
k_list = np.arange(0, 50000, 100)
eps = 1e-3
img_size = 256
batch_size = 1000
MAX_RESOLUTION = img_size
PIXEL_SIZE = 1.3
dtype = np.float32
# Create a source object for 2D images
print(f'Read in images from {STARFILE_IN} and preprocess the images.')
source = RelionSource(
    STARFILE_IN,
    DATA_FOLDER,
    pixel_size=PIXEL_SIZE,
    max_rows=MAX_ROWS,
    n_workers=60
)

print(len(source.unique_filters))
print(f'Set the resolution to {MAX_RESOLUTION} X {MAX_RESOLUTION}')
if MAX_RESOLUTION < source.L:
    source.downsample(MAX_RESOLUTION)

fle = FLEBasis2D(img_size, img_size, eps=eps, dtype=dtype)

options = {
    "whiten": True,
    "noise_psd": None,
    "radius": 0.9,
    "single_pass": True,
    "flip_sign": True,
    "single_whiten_filter": False,
    "batch_size": batch_size,
    "dtype": dtype,
    "correct_contrast": True,
    "subtract_background": True,
}

fast_pca = FastPCA(source, fle, options)

denoise_options = {
    "denoise_df_id": k_list,
    # not specifying "denoise_df_num" means use all images in those defocus groups
    "return_denoise_error": False,
    "store_images": True,
}


results = fast_pca.denoise_images(denoise_options=denoise_options)
np.save('your directory/output/10081/imgs_est.npy', results["denoised_images"])
np.save('your directory/output/10081/img_idx_list.npy', results["image_indices_list"])
np.save('your directory/output/10081/denoise_idx_start.npy', results["image_indices_start"])
np.save('your directory/output/10081/denoise_idx_num.npy', results["image_indices_number"])
np.save('your directory/output/10081/covar_est.npy', fast_pca.covar_est)
np.save('your directory/output/10081/contrast_est.npy', fast_pca.contrast_est)


print("Done")
