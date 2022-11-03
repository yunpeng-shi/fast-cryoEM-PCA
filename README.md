# Fast Principal Component Analysis for Cryo-EM Images

This repository includes the code for implementing the method of the paper [Fast Principal Component Analysis for Cryo-EM Images, arXiv preprint, 2022](http://arxiv.org/abs/2210.17501).

Our code provides a fast implementation of covariance estimation based on the recent Fourier-Bessel expansion method [FLE](https://github.com/nmarshallf/fle_2d). Based on the covariance estimation, it offers a fast solution of joint CTF correction and denoisng of cryo-EM images. As an option, it can also estimate amplitude contrasts of individual images, which for the first time is available in the ab-initio reconstruction stage.




