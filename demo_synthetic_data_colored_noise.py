"""
Demo Code for Fast PCA of Cryo-EM Images with colored noise
"""


import utils_cwf_fast_batch as utils
import matplotlib.pyplot as plt
import mrcfile
from fle_2d_single import FLEBasis2D
from fast_pca import FastPCA
import logging
import os
import numpy as np
import scipy.linalg as LA
from aspire.source.simulation import Simulation
from aspire.volume import Volume
from aspire.operators import RadialCTFFilter, ArrayFilter

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "./data/synthetic")  # the directory of the clean volume data


#######################################
## Parameters
#######################################
img_size = 64
defocus_ct = 100  # the number of defocus groups
sn_ratio = 1
logger.info(f"Signal to noise ratio is {sn_ratio}.")

batch_size = 1000
num_imgs = 10000
logger.info(f"image num {num_imgs}")

eps = 1e-3
dtype = np.float32


# Specify the CTF parameters
pixel_size = 5 * 65 / img_size # Pixel size of the images (in angstroms)

voltage = 200  # Voltage (in KV)
defocus_min = 1e4  # Minimum defocus value (in angstroms)
defocus_max = 3e4  # Maximum defocus value (in angstroms)

Cs = 2.0  # Spherical aberration
alpha = 0.1  # Amplitude contrast

# create CTF indices for each image, e.g. h_idx[0] returns the CTF index (0 to 99 if there are 100 CTFs) of the 0-th image
h_idx = utils.create_ordered_filter_idx(num_imgs, defocus_ct)


logger.info(f"Simulation running in {dtype} precision.")

logger.info("Initialize simulation object and CTF filters.")
# Create filters. This is a list of CTFs. Each element corresponds to a UNIQUE CTF
h_ctf = [
    RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]

# Load the map file of a 70S Ribosome
logger.info(
    f"Load 3D map and downsample 3D map to desired grids "
    f"of {img_size} x {img_size} x {img_size}."
)
infile = mrcfile.open(os.path.join(DATA_DIR, "clean70SRibosome_vol_65p.mrc"))

# We prefer that our various arrays have consistent dtype.
vols = Volume(infile.data.astype(dtype) / np.max(infile.data))
vols = vols.downsample(img_size)

# Create a simulation object with specified filters and the downsampled 3D map
logger.info("Use downsampled map to creat simulation object.")

# this is for generating CTF-affected clean projections.
# We use this to determine the noise variance so that our simulated images have targeted SNR

source_ctf_clean = Simulation(
    L=img_size,
    n=num_imgs,
    vols=vols,
    offsets=0.0,
    amplitudes=1.0,
    unique_filters=h_ctf,
    filter_indices=h_idx,
    dtype=dtype,
)

# determine noise variance to create noisy images with certain SNR
noise_var0 = utils.get_noise_var_batch(source_ctf_clean, sn_ratio, batch_size)

logger.info(f"Signal to noise ratio is {sn_ratio}.")
rad_mat = utils.compute_radius_mat(img_size)
psd_fun = lambda x: 1/(x * img_size / 20 + 1)  # specify noise psd as a radial function
im_psd = psd_fun(rad_mat)
mean_psd = np.mean(im_psd)
im_psd = im_psd * noise_var0 / mean_psd  # adjust noise level to the specified SNR
noise_filter = ArrayFilter(im_psd)

# simulation source with CTF and noise (after determining noise var)

source = Simulation(
    L=img_size,
    n=num_imgs,
    vols=vols,
    unique_filters=h_ctf,
    filter_indices=h_idx,
    offsets=0.0,
    amplitudes=1.0,
    dtype=dtype,
    noise_filter=noise_filter,
)

# Fourier-Bessel expansion object
fle = FLEBasis2D(img_size, img_size, eps=eps, dtype=dtype)
r = (fle.pts.reshape(1, -1)) * fle.h / np.pi
# ground truth radial noise psd at sampled 1-D grid points
radial_psd_gt = psd_fun(r) / mean_psd * noise_var0

# get clean sample mean and covariance
mean_clean = utils.get_clean_mean_batch(source, fle, batch_size)
covar_clean = utils.get_clean_covar_batch(source, fle, mean_clean, batch_size, dtype)



options = {
    "whiten": True,  # whiten the noise
    "noise_psd": radial_psd_gt,
    #"noise_psd": None,   # noise_psd will be estimated if use this line
    "store_noise_psd": False,
    "radius": 0.9, # estimate the noise psd from corners of images outside of the mask with radius 0.9 (max is 1)
    "single_pass": True, # whether estimate mean and covariance together (single pass over data), not separately
    "single_whiten_filter": False, # whiten images by defocus groups (not a single filter)
    "batch_size": batch_size,
    "dtype": dtype
}

# create fast PCA object
fast_pca = FastPCA(source, fle, options)

denoise_options = {
    "denoise_df_id": [0, 30, 60, 90], # denoise 0-th, 30-th, 60-th, 90-th defocus groups
    "denoise_df_num": [10, 15, 1, 240], # for each defocus group, respectively denoise the first 10, 15, 1, 100 images
                                        # 240 exceed the number of images (100) per defocus group, so only 100 images will be returned
    "return_denoise_error": True,
    "store_images": True,
}

# code example 1: run in three steps, two passes over data for covariance estimation
# mean_est = fast_pca.estimate_mean()
# _, covar_est = fast_pca.estimate_mean_covar(mean_est=mean_est)
# results = fast_pca.denoise_images(mean_est=mean_est, covar_est=covar_est, denoise_options=denoise_options)

# code example 2: single pass over data (combine mean and covariance estimation)
# mean_est, covar_est = fast_pca.estimate_mean_covar()
# results = fast_pca.denoise_images(mean_est=mean_est, covar_est=covar_est, denoise_options=denoise_options)

# code example 3: Combine all steps in one-line of code (equivalent to step 2)
results = fast_pca.denoise_images(denoise_options=denoise_options)

err_denoise = results["mean_denoise_error"]
imgs_gt = results["clean_images"]
imgs_raw = results["raw_images"]
imgs_est = results["denoised_images"]
mean_est = fast_pca.mean_est
covar_est = fast_pca.covar_est

err_mean = LA.norm(mean_clean-mean_est)/LA.norm(mean_clean)
_, err_covar = utils.compute_covar_err(covar_est, covar_clean)
_, frc_vec = utils.compute_frc(imgs_est, imgs_gt, fle.n_angular, fle.eps, dtype=dtype)

print(f'error of mean estimation = {err_mean}')
print(f'error of covar estimation = {err_covar}')
print(f'error of image restoration = {err_denoise}')
im = 0
plt.subplot(2,2,1)
plt.imshow(imgs_gt[im], 'gray')
plt.title("clean")
plt.axis("off")
plt.subplot(2,2,2)
plt.imshow(imgs_raw[im], 'gray')
plt.title("noisy")
plt.axis("off")
plt.subplot(2,2,3)
plt.imshow(imgs_est[im], 'gray')
plt.title("denoised")
plt.axis("off")
plt.subplot(2,2,4)
plt.plot(frc_vec, '-r')
plt.title("FRC")

plt.show()

print("Done")
