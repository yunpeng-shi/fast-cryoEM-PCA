# Fast Principal Component Analysis for Cryo-EM Images

This repository includes the code for implementing the method of the paper [Fast Principal Component Analysis for Cryo-EM Images, arXiv preprint, 2022](http://arxiv.org/abs/2210.17501).

Our code provides a fast implementation of covariance estimation based on the recent Fourier-Bessel expansion method [FLE](https://github.com/nmarshallf/fle_2d). Based on the covariance estimation, it offers a fast, ab-initio and unsupervised method to jointly correct CTFs  and denoise the cryo-EM images. As an option, it can also estimate amplitude contrasts of individual images, which for the first time is available in the ab-initio reconstruction stage. The method is very fast compared to the existing ones, especially when there are many distinct CTFs. See below for runtime comparison and examples of denoised experimental images.

<img src="https://github.com/yunpeng-shi/fast-cryoEM-PCA/blob/main/time.png" width="500" height="200">
<img src="https://github.com/yunpeng-shi/fast-cryoEM-PCA/blob/main/denoise.png" width="500" height="200">

## Installation

Our code relies on two packages: [``ASPIRE-python``](https://github.com/ComputationalCryoEM/ASPIRE-Python) for single particle reconstruction and [``FLE``](https://github.com/nmarshallf/fle_2d) for fast Fourier-Bessel expansion. 



