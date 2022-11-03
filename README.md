# Fast Principal Component Analysis for Cryo-EM Images

This repository includes the code for implementing the method of the paper [Fast Principal Component Analysis for Cryo-EM Images, arXiv preprint, 2022](http://arxiv.org/abs/2210.17501).

Our code provides a fast implementation of covariance estimation based on the recent Fourier-Bessel expansion method [FLE](https://github.com/nmarshallf/fle_2d). Based on the covariance estimation, it offers a fast, ab-initio and unsupervised method to jointly correct CTFs  and denoise the cryo-EM images. As an option, it can also estimate amplitude contrasts of individual images, which for the first time is available in the ab-initio reconstruction stage. The method is very fast compared to the existing ones, especially when there are many distinct CTFs. See below for runtime comparison and examples of denoised experimental images.

<img src="https://github.com/yunpeng-shi/fast-cryoEM-PCA/blob/main/time.png" width="500" height="200">
<img src="https://github.com/yunpeng-shi/fast-cryoEM-PCA/blob/main/denoise.png" width="500" height="180">

## Installation

Our code relies on two packages: [``ASPIRE-python``](https://github.com/ComputationalCryoEM/ASPIRE-Python) for single particle reconstruction and [``FLE``](https://github.com/nmarshallf/fle_2d) for fast Fourier-Bessel expansion. Please see the two repositories for the installation details. [``FLE``](https://github.com/nmarshallf/fle_2d)  only allows double precision operations for higher accuracy and better numerical stability. Alternatively, one can use ``fle_2d_single.py`` provided in this repository for single precision operations, though it's not expected to be as accurate as [``FLE``](https://github.com/nmarshallf/fle_2d). To use ``fle_2d_single.py``, one should put it in the same directory of the file [``jn_zeros_n=3000_nt=2500.mat``](https://github.com/nmarshallf/fle_2d/blob/main/src/fle_2d/jn_zeros_n%3D3000_nt%3D2500.mat).


## Tutorial Code

After installing all the dependencies, run demo code [``demo_synthetic_data_white_noise.py``](https://github.com/yunpeng-shi/fast-cryoEM-PCA/blob/main/demo_synthetic_data_white_noise.py) and [``demo_synthetic_data_colored_noise.py``](https://github.com/yunpeng-shi/fast-cryoEM-PCA/blob/main/demo_synthetic_data_colored_noise.py) for synthetic simulations with white and colored noise respectively. To run demo code [``demo_experimental_data_10081.py``](https://github.com/yunpeng-shi/fast-cryoEM-PCA/blob/main/demo_experimental_data_10081.py) for experimental data, download picked particle data from [EMPIAR-10081](https://www.ebi.ac.uk/empiar/EMPIAR-10081/) (use the one of size 13.7 GB) to the directory ``./data/Particles/micrographs/``, and use the star file provided in ``./data/particle_stacks/``, not the one on [EMPIAR-10081](https://www.ebi.ac.uk/empiar/EMPIAR-10081/).



