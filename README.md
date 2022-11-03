# Fast Principal Component Analysis for Cryo-EM Images

This repository includes the code for implementing the method of the paper [Fast Principal Component Analysis for Cryo-EM Images, arXiv preprint, 2022](http://arxiv.org/abs/2210.17501).

Our code provides a fast implementation of covariance estimation based on the recent Fourier-Bessel expansion method [FLE](https://github.com/nmarshallf/fle_2d). Based on the covariance estimation, it offers a fast, ab-initio and unsupervised method to jointly correct CTFs  and denoise the cryo-EM images. As an option, it can also estimate amplitude contrasts of individual images, which for the first time is available in the ab-initio reconstruction stage. The method is very fast compared to the existing ones, especially when there are many distinct CTFs. See below for runtime comparison and examples of denoised experimental images.

<img src="https://github.com/yunpeng-shi/fast-cryoEM-PCA/blob/main/time.png" width="500" height="200">
<img src="https://github.com/yunpeng-shi/fast-cryoEM-PCA/blob/main/denoise.png" width="500" height="180">

## Installation

Our code relies on two packages: [``ASPIRE-python``](https://github.com/ComputationalCryoEM/ASPIRE-Python) for single particle reconstruction and [``FLE``](https://github.com/nmarshallf/fle_2d) for fast Fourier-Bessel expansion. Please see the two repositories for the installation details. [``FLE``](https://github.com/nmarshallf/fle_2d)  only allows double precision operations for higher accuracy and better numerical stability. Alternatively, one can use ``fle_2d_single.py`` provided in this repository for single precision operations, though it's not expected to be as accurate as [``FLE``](https://github.com/nmarshallf/fle_2d). To use ``fle_2d_single.py``, one should put it in the same directory of the file [``jn_zeros_n=3000_nt=2500.mat``](https://github.com/nmarshallf/fle_2d/blob/main/src/fle_2d/jn_zeros_n%3D3000_nt%3D2500.mat).


## Tutorial Code

After installing all the dependencies, run demo code [``demo_synthetic_data_white_noise.py``](https://github.com/yunpeng-shi/fast-cryoEM-PCA/blob/main/demo_synthetic_data_white_noise.py) and [``demo_synthetic_data_colored_noise.py``](https://github.com/yunpeng-shi/fast-cryoEM-PCA/blob/main/demo_synthetic_data_colored_noise.py) for synthetic simulations with white and colored noise respectively. To run demo code [``demo_experimental_data_10081.py``](https://github.com/yunpeng-shi/fast-cryoEM-PCA/blob/main/demo_experimental_data_10081.py) for experimental data, download picked particle data from [EMPIAR-10081](https://www.ebi.ac.uk/empiar/EMPIAR-10081/) (use the one of size 13.7 GB) to the directory ``./data/Particles/micrographs/``, and use the star file provided in ``./data/particle_stacks/``, not the one on [EMPIAR-10081](https://www.ebi.ac.uk/empiar/EMPIAR-10081/).

## A Variety of Features

Our covariance solver offers a variety of options:

```
default_options = {
    "whiten": True,
    "noise_psd": None,
    "store_noise_psd": True,
    "noise_var": None,
    "radius": 0.9,
    "batch_size": 1000,
    "single_pass": True,
    "single_whiten_filter": False,
    "flip_sign": False,
    "correct_contrast": False,
    "subtract_background": False,
    "dtype": np.float64,
    "verbose": True
}
```
Here we briefly explain each one:

``whiten``: whether or not whiten the image

``noise_psd``: if choose ``None``, then noise power spectrum (radial functions) will be estimated. The radial function representation largely reduces the memory requirement, and it also reduces the noise in psd estimation by averaging over the rings (psd on the rings are estimated by NUFFT).

``store_noise_psd``: whether stores noise psd (1-D radial functions)

``noise_var``: noise variance, automatically set as 1 if whiten is true

``radius``: radius of mask used to estimate noise psd and background mean/variance. maximum value is 1.

``batch_size``: batch size for loading image data; useful when the number of images is large.

``single_pass``: whether estimate mean and covariance together (so only one pass over data), otherwise estimate mean and then covariance (two passes over data)

``single_whiten_filter``: whether use only one whiten filter for all images. If choose ``False``, then whiten image by defocus groups.

``flip_sign``: whether flip sign when loading raw images. A common choice is ``True`` for raw experimental images.

``correct_contrast``: whether correct and estimate image amplitude contrast.

``subtract_background``: whether estimate and subtract background mean

``dtype``: single or double precision

``verbose``: whether output progress when running code

