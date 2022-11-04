import numpy as np
from aspire.operators import BlkDiagMatrix
from utils_cwf_fast_batch import estimate_radial_psd, get_batch_idx, get_sample_covar, shrink_covar_backward, \
    wiener_filter, estimate_bg
from aspire.optimization import fill_struct
from aspire.utils import make_psd
import scipy.linalg as LA
import time
import logging

logger = logging.getLogger(__name__)


class FastPCA:

    def __init__(self, src, basis=None, options=None):
        self.src = src
        self.basis = basis
        self.options = options
        self.mean_est = None
        self.covar_est = None
        self._build()

    def _build(self):

        if self.basis is None:
            from fle_2d_new import FLEBasis2D
            self.basis = FLEBasis2D(self.src.L, self.src.L, eps=1e-3)

        self.h_ctf = self.src.unique_filters
        self.h_idx = self.src.filter_indices
        self.pixel_size = self.h_ctf[0].pixel_size
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

        self.options = fill_struct(self.options, default_options)

        self.whiten = self.options["whiten"]
        self.noise_psd = self.options["noise_psd"]
        self.store_psd = self.options["store_noise_psd"]
        self.noise_var = self.options["noise_var"]
        self.single_pass = self.options["single_pass"]
        self.single_filter = self.options["single_whiten_filter"]
        self.flip_sign = self.options["flip_sign"]
        self.correct_contrast = self.options["correct_contrast"]
        self.verbose = self.options["verbose"]
        self.subtract_background = self.options["subtract_background"]

        if self.whiten:
            if self.verbose:
                logger.info(f"images will be whitened")
            self.noise_var = 1
            if self.noise_psd is not None:
                if self.verbose:
                    logger.info(f"noise psd exists... store psd... no need to estimate")
                self.store_psd = True
                assert self.noise_psd.shape[-1] == self.basis.pts.shape[0] or self.noise_psd.shape[0] == \
                       self.basis.pts.shape[0]
                if np.prod(self.noise_psd.shape) == self.basis.pts.shape[0]:
                    if self.verbose:
                        logger.info(f"will use a single whiten filter for all images")
                    self.noise_psd = self.noise_psd.reshape(1, -1)
                else:
                    assert self.noise_psd.shape[0] == len(self.h_ctf)
                    if self.single_filter:
                        if self.verbose:
                            logger.info(f"use the same whiten filter for all...")
                        self.noise_psd = np.mean(self.noise_psd, axis=0).reshape(1, -1)

                self.estimate_noise = False
            else:
                if self.verbose:
                    logger.info(f"noise psds will be estimated")
                self.estimate_noise = True
                if self.single_filter or self.single_pass:
                    self.store_psd = True
                    if self.verbose:
                        logger.info(f"noise psds will be stored...")

        else:
            if self.verbose:
                logger.info(f"noise is already white")
            self.noise_psd = None
            assert self.noise_var is not None
            self.estimate_noise = False
            self.store_psd = False

        self.bgRadius = self.options["radius"]
        self.batch_size = self.options["batch_size"]
        self.dtype = self.options["dtype"]

        self.options = {
            "whiten": self.whiten,
            "noise_psd": self.noise_psd,
            "store_noise_psd": self.store_psd,
            "noise_var": self.noise_var,
            "radius": self.bgRadius,
            "batch_size": self.batch_size,
            "single_pass": self.single_pass,
            "single_whiten_filter": self.single_filter,
            "correct_contrast": self.correct_contrast,
            "flip_sign": self.flip_sign,
            "subtract_background": self.subtract_background,
            "dtype": self.dtype,
            "verbose": self.verbose
        }

        self.t_basis_expansion = 0
        self.t_ctf_expansion = 0
        self.t_cov_est = 0
        self.t_denoise = 0
        self.t_whiten = 0
        self.t_bg = 0
        self.t_load = 0

    def estimate_mean_covar(self, mean_est=None):
        basis = self.basis
        mean_num = 0
        mean_deno = 0
        partition = []
        num_imgs = self.src.n
        h_ctf = self.src.unique_filters
        h_idx = self.src.filter_indices
        n_blk = basis.n_blk
        blk_size = basis.blk_size
        blk_ind = basis.blk_ind

        if mean_est is not None:
            if self.verbose:
                logger.info(f"image mean already exists... no need to estimate...")
            self.single_pass = False
            self.mean_est = mean_est
        else:
            if self.verbose:
                logger.info(f"need to estimate the mean...")
            self.mean_est = np.zeros((basis.ne,))

        _, unique_count_all = np.unique(h_idx, return_counts=True)
        batch_start, batch_size_list, batch_num = get_batch_idx(num_imgs, self.batch_size)
        for ell in range(n_blk):
            partition.append([blk_size[ell], blk_size[ell]])

        self.covar_est = BlkDiagMatrix.zeros(partition, dtype=self.dtype)
        B_mat = BlkDiagMatrix.zeros(partition, dtype=self.dtype)
        noise_mat = BlkDiagMatrix.zeros(partition, dtype=self.dtype)
        L_mat = BlkDiagMatrix.zeros(partition, dtype=self.dtype)

        if self.whiten:
            tw0 = time.time()
            if self.single_filter and self.estimate_noise:
                if self.verbose:
                    logger.info(f"estimate a single noise psd...")
                self.noise_psd = np.zeros((1, basis.pts.shape[0]))
                for l in range(0, batch_num):
                    weights = batch_size_list[l] / num_imgs

                    tld0 = time.time()

                    if self.flip_sign:
                        imgs_noise_l = -self.src.images(start=batch_start[l], num=batch_size_list[l]).asnumpy()
                    else:
                        imgs_noise_l = self.src.images(start=batch_start[l], num=batch_size_list[l]).asnumpy()

                    tld1 = time.time()
                    self.t_load += tld1 - tld0

                    if self.subtract_background:
                        _, std_l = estimate_bg(imgs_noise_l)
                        imgs_noise_l = imgs_noise_l / std_l

                    radial_psd_im = estimate_radial_psd(imgs_noise_l, basis, self.bgRadius, self.dtype)
                    self.noise_psd += weights * np.mean(radial_psd_im, axis=0)
                # else:
                #     self.noise_psd = np.mean(self.noise_psd, axis=0).reshape(1, -1)

                self.estimate_noise = False

            elif self.store_psd and self.estimate_noise:

                if self.verbose:
                    logger.info(f"estimate noise psds for each defocus group...")
                self.noise_psd = np.zeros((len(h_ctf), basis.pts.shape[0]))
                for l in range(0, batch_num):

                    tld0 = time.time()

                    if self.flip_sign:
                        imgs_noise_l = -self.src.images(start=batch_start[l], num=batch_size_list[l]).asnumpy()
                    else:
                        imgs_noise_l = self.src.images(start=batch_start[l], num=batch_size_list[l]).asnumpy()

                    tld1 = time.time()
                    self.t_load += tld1 - tld0

                    if self.subtract_background:
                        _, std_l = estimate_bg(imgs_noise_l)
                        imgs_noise_l = imgs_noise_l / std_l

                    h_idx_l = h_idx[batch_start[l]:batch_start[l + 1]]
                    unique_val, unique_ind, unique_count = np.unique(h_idx_l, return_inverse=True,
                                                                     return_counts=True)
                    radial_psd_im = estimate_radial_psd(imgs_noise_l, basis, self.bgRadius, self.dtype)

                    for k_ind in range(len(unique_val)):
                        k = unique_val[k_ind]
                        weights = unique_count[k_ind] / unique_count_all[k]
                        self.noise_psd[k, :] += weights * np.mean(radial_psd_im[h_idx_l == k, :], axis=0)

                self.estimate_noise = False

            if self.noise_psd is not None:
                self.estimate_noise = False

            tw1 = time.time()
            self.t_whiten += tw1 - tw0

        if not self.single_pass:

            if mean_est is None:
                if self.verbose:
                    logger.info(f"two passes over the data for covariance estimation...")
                    logger.info(f"the first pass is for mean estimation...")
                    logger.info(f"start mean estimation...")
                self.mean_est = self.estimate_mean()
                if self.verbose:
                    logger.info(f"mean estimation completed...")
        else:
            if self.verbose:
                logger.info(f"ONLY one pass over the data for covariance estimation...")
            b0_mean = np.zeros((len(h_ctf), blk_size[0]))

        if self.verbose:
            logger.info(f"start covariance estimation...")
        for l in range(0, batch_num):
            if self.verbose:
                logger.info(f"drawing {l}-th batch of images")
            weights = batch_size_list[l] / num_imgs

            tld0 = time.time()

            if self.flip_sign:
                imgs_noise_l = -self.src.images(start=batch_start[l], num=batch_size_list[l]).asnumpy()
            else:
                imgs_noise_l = self.src.images(start=batch_start[l], num=batch_size_list[l]).asnumpy()

            tld1 = time.time()
            self.t_load += tld1 - tld0

            if self.subtract_background:
                tbg0 = time.time()
                mean_l, std_l = estimate_bg(imgs_noise_l)
                tbg1 = time.time()
                self.t_bg += tbg1 - tbg0

                if self.verbose:
                    logger.info(f"expanding the images into FFB basis")
                    logger.info(f"subtracting the background ...")

                imgs_noise_l_bg = (imgs_noise_l - mean_l) / std_l

                if self.verbose:
                    logger.info(f"finished background subtraction")

                tbe0 = time.time()
                coeffs_eig_l = basis.evaluate_t(imgs_noise_l_bg)
                tbe1 = time.time()
                self.t_basis_expansion += tbe1 - tbe0
                if self.verbose:
                    logger.info(f"basis expansion finished")

            else:

                if self.verbose:
                    logger.info(f"expanding the images into FFB basis")
                tbe0 = time.time()
                coeffs_eig_l = basis.evaluate_t(imgs_noise_l)
                tbe1 = time.time()
                self.t_basis_expansion += tbe1 - tbe0
                if self.verbose:
                    logger.info(f"basis expansion finished")


            coeffs_l = basis.to_angular_order(coeffs_eig_l.T).T
            h_idx_l = h_idx[batch_start[l]:batch_start[l + 1]]
            unique_val, unique_ind, unique_count = np.unique(h_idx_l, return_inverse=True, return_counts=True)
            unique_val = unique_val.astype(int)
            voltage_list = np.array([h_ctf[k].voltage for k in unique_val])
            cs_list = np.array([h_ctf[k].Cs for k in unique_val])
            alpha_list = np.array([h_ctf[k].alpha for k in unique_val])
            defocus_list = np.array([h_ctf[k].defocus_mean for k in unique_val])


            tce0 = time.time()

            rwts_mat_l = basis.expand_ctf(voltage_list, cs_list, alpha_list, defocus_list, self.pixel_size)

            tce1 = time.time()
            self.t_ctf_expansion += tce1 - tce0
            rwts_mat_l = basis.to_angular_order(rwts_mat_l.T).T

            if self.whiten:

                tw0 = time.time()

                if self.single_pass:
                    if self.noise_psd.shape[0] > 1:
                        whiten_radial = 1 / np.sqrt(self.noise_psd[unique_val, :])
                    else:
                        whiten_radial = 1 / np.sqrt(self.noise_psd)

                    whiten_fb_eig = basis.expand_raidal_vec(whiten_radial)
                    whiten_fb = basis.to_angular_order(whiten_fb_eig.T).T

                else:

                    if self.estimate_noise:
                        if self.verbose:
                            logger.info(f"estimate and expand whiten filters in to FFB basis...two passes...")

                        if self.subtract_background:
                            radial_psd_im = estimate_radial_psd(imgs_noise_l / std_l, basis, self.bgRadius)
                        else:
                            radial_psd_im = estimate_radial_psd(imgs_noise_l, basis, self.bgRadius, self.dtype)
                        radial_psd_df = np.zeros((len(unique_val), basis.pts.shape[0]))
                        for k_ind in range(len(unique_val)):
                            k = unique_val[k_ind]
                            radial_psd_df[k_ind, :] = np.mean(radial_psd_im[h_idx_l == k, :], axis=0)

                    elif self.noise_psd.shape[0] == 1:
                        radial_psd_df = self.noise_psd
                    else:
                        radial_psd_df = self.noise_psd[unique_val, :]

                    # whiten_radial = np.zeros((num_whiten, basis.pts.shape[0]))
                    whiten_radial = 1 / np.sqrt(radial_psd_df)

                    whiten_fb_eig = basis.expand_raidal_vec(whiten_radial)
                    whiten_fb = basis.to_angular_order(whiten_fb_eig.T).T

                rwts_mat_l = whiten_fb * rwts_mat_l
                if whiten_fb.shape[0] == 1:
                    coeffs_l = whiten_fb * coeffs_l
                else:
                    coeffs_l = whiten_fb[unique_ind, :] * coeffs_l

                tw1 = time.time()
                self.t_whiten += tw1 - tw0

            t_cov0 = time.time()

            weights_vec = (unique_count / batch_size_list[l]).reshape(-1, 1)
            coeffs_ctf_l = rwts_mat_l[unique_ind, :] * coeffs_l
            # covariance estimation
            if not self.single_pass:
                B_mat += weights * get_sample_covar((rwts_mat_l[unique_ind] ** 2) * self.mean_est, coeffs_ctf_l, basis,
                                                    self.dtype)
            else:

                rwts_mat_l0 = rwts_mat_l[:, blk_ind[0]:blk_ind[1]]
                coeffs_ctf_l0 = rwts_mat_l0[unique_ind, :] * coeffs_l[:, blk_ind[0]:blk_ind[1]]
                # mean estimation
                mean_num += weights * np.mean(coeffs_ctf_l0, 0)
                rwts_mat_l0_weighted = np.sqrt(weights_vec) * rwts_mat_l0
                mean_deno += weights * np.sum(rwts_mat_l0_weighted ** 2, 0)

                B_mat += weights * get_sample_covar(np.zeros((basis.ne,)), coeffs_ctf_l, basis, self.dtype)

                for k_ind in range(len(unique_val)):
                    k = unique_val[k_ind]
                    coeffs_ctf_k0 = coeffs_ctf_l[h_idx_l == k, blk_ind[0]:blk_ind[1]]
                    b0_mean[k] += np.mean(coeffs_ctf_k0, axis=0) * coeffs_ctf_k0.shape[0] / num_imgs

            for ell in range(0, n_blk):
                r_k2 = rwts_mat_l[:, blk_ind[ell]:blk_ind[ell + 1]] ** 2
                wr_k2 = np.sqrt(weights_vec) * r_k2
                L_mat[ell] += weights * np.sum((wr_k2[:, :, None] * wr_k2[:, None]), axis=0)
                wr_k2 = weights_vec * r_k2
                noise_mat[ell] += weights * np.diag(np.sum(wr_k2, axis=0))

            t_cov1 = time.time()
            self.t_cov_est += t_cov1 - t_cov0

        if self.single_pass:

            self.mean_est[blk_ind[0]:blk_ind[1]] = mean_num / mean_deno

            unique_val, unique_ind, unique_count = np.unique(h_idx, return_inverse=True, return_counts=True)
            weights_vec = (unique_count / num_imgs).reshape(-1, 1)
            ctf_batch_start, ctf_batch_size_list, ctf_batch_num = get_batch_idx(len(unique_val), self.batch_size)


            for l in range(ctf_batch_num):


                unique_val_l = unique_val[ctf_batch_start[l]: ctf_batch_start[l + 1]]
                unique_val_l = unique_val_l.astype(int)
                voltage_list = np.array([h_ctf[k].voltage for k in unique_val_l])
                cs_list = np.array([h_ctf[k].Cs for k in unique_val_l])
                alpha_list = np.array([h_ctf[k].alpha for k in unique_val_l])
                defocus_list = np.array([h_ctf[k].defocus_mean for k in unique_val_l])


                tce0 = time.time()

                rwts_mat_l = basis.expand_ctf(voltage_list, cs_list, alpha_list, defocus_list, self.pixel_size)

                tce1 = time.time()
                self.t_ctf_expansion += tce1 - tce0
                rwts_mat_l = basis.to_angular_order(rwts_mat_l.T).T

                if self.whiten:
                    if self.verbose:
                        logger.info(f"expanding whiten filters in to FFB basis...single pass")
                    tw0 = time.time()
                    if self.noise_psd.shape[0] > 1:
                        whiten_radial = 1 / np.sqrt(self.noise_psd[unique_val_l, :])
                    else:
                        whiten_radial = 1 / np.sqrt(self.noise_psd)

                    whiten_fb_eig = basis.expand_raidal_vec(whiten_radial)
                    whiten_fb = basis.to_angular_order(whiten_fb_eig.T).T
                    rwts_mat_l = whiten_fb * rwts_mat_l

                    tw1 = time.time()
                    self.t_whiten += tw1 - tw0

                t_cov0 = time.time()

                rwts_mat_l0 = rwts_mat_l[:, blk_ind[0]:blk_ind[1]]
                weights_l = weights_vec[ctf_batch_start[l]: ctf_batch_start[l + 1]]
                mean_coeff_l = (rwts_mat_l0 ** 2) * self.mean_est[blk_ind[0]:blk_ind[1]]
                b0_mean_l = b0_mean[ctf_batch_start[l]: ctf_batch_start[l + 1]]
                B_mat[0] -= np.sum(mean_coeff_l[:, :, None] * b0_mean_l[:, None], axis=0)
                B_mat[0] -= np.sum(b0_mean_l[:, :, None] * mean_coeff_l[:, None], axis=0)
                B_mat[0] += np.sum((weights_l * mean_coeff_l)[:, :, None] * mean_coeff_l[:, None], axis=0)
                t_cov1 = time.time()
                self.t_cov_est += t_cov1 - t_cov0

        if self.verbose:
            logger.info(f"eigenvalue shrinkage for covariance estimation")

        t_cov0 = time.time()

        B_mat_shrink = shrink_covar_backward(
            b=B_mat,
            b_noise=noise_mat,
            n=num_imgs,
            noise_var=self.noise_var,
            shrinker="operator_norm",
            dtype=self.dtype
        )

        B_mat_shrink = B_mat_shrink.make_psd()
        self.B = B_mat_shrink
        self.L = L_mat
        for ell in range(0, n_blk):
            self.covar_est[ell] = B_mat_shrink[ell] / L_mat[ell]

        self.covar_est = self.covar_est.make_psd()

        if self.correct_contrast:
            self.covar_est = self.contrast_correction()

        t_cov1 = time.time()
        self.t_cov_est += t_cov1 - t_cov0
        if self.verbose:
            logger.info(f"covariance estimation completed!")

        self.options["noise_psd"] = self.noise_psd
        if self.noise_psd is None:
            self.estimate_noise = True
        else:
            self.estimate_noise = False

        return self.mean_est, self.covar_est

    def denoise_images(self, mean_est=None, covar_est=None, denoise_options=None):

        if mean_est is None or covar_est is None:
            mean_est, covar_est = self.estimate_mean_covar()

        if self.noise_psd is None:
            self.estimate_noise = True
        else:
            self.estimate_noise = False

        default_denoise_options = {
            "denoise_df_id": np.arange(0, len(self.h_ctf)),
            "denoise_df_num": None,
            "return_denoise_error": False,
            "store_images": False,
        }
        denoise_options = fill_struct(denoise_options, default_denoise_options)
        self.options = fill_struct(self.options, denoise_options)
        basis = self.basis
        img_size = self.src.L
        h_ctf = self.src.unique_filters
        h_idx = self.src.filter_indices


        denoise_df_id = self.options["denoise_df_id"]
        denoise_df_num = self.options["denoise_df_num"]
        denoise_df_ct = len(denoise_df_id)
        store_images = self.options["store_images"]
        return_error = self.options["return_denoise_error"]

        denoise_idx_start = np.zeros((denoise_df_ct,))
        denoise_idx_num = np.zeros((denoise_df_ct,))
        for k in range(denoise_df_ct):
            k_idx = np.where(h_idx == denoise_df_id[k])[0]
            k_start = k_idx[0]
            k_num = np.argmax(h_idx[k_start:] != denoise_df_id[k])
            if k_num == 0:
                k_num = len(h_idx) - k_start
            denoise_idx_start[k] = k_start
            denoise_idx_num[k] = k_num

        denoise_idx_start = denoise_idx_start.astype(int)
        if denoise_df_num is not None:
            denoise_idx_num = (np.minimum(denoise_idx_num, denoise_df_num)).astype(int)

        num_denoise_imgs = np.sum(denoise_idx_num).astype(int)
        err_denoise_vec = np.zeros((num_denoise_imgs,))

        if self.correct_contrast:
            self.contrast_est = np.zeros((num_denoise_imgs,))
            im_one = np.ones((self.src.L, self.src.L))
            coeff_one = self.basis.to_angular_order(self.basis.evaluate_t(im_one)).flatten()
            blk_ind = self.basis.blk_ind
            coeff_one0 = coeff_one[blk_ind[0]:blk_ind[1]]

        whiten_radial_full = None

        if self.whiten:
            if not self.estimate_noise:
                whiten_radial_full = 1 / np.sqrt(self.noise_psd)

            else:
                if self.single_filter:
                    whiten_radial_full = np.zeros((1, basis.pts.shape[0]))
                elif self.single_pass:
                    whiten_radial_full = np.zeros((denoise_df_ct, basis.pts.shape[0]))

                for k in range(denoise_df_ct):
                    k_batch_start, k_batch_size_list, k_batch_num = get_batch_idx(denoise_idx_num[k], self.batch_size)
                    k_batch_start += denoise_idx_start[k]

                    for l in range(k_batch_num):
                        weights = k_batch_size_list[l] / num_denoise_imgs
                        weights_k = k_batch_size_list[l] / denoise_idx_num[k]

                        if self.flip_sign:
                            imgs_noise_l = -self.src.images(start=k_batch_start[l], num=k_batch_size_list[l]).asnumpy()
                        else:
                            imgs_noise_l = self.src.images(start=k_batch_start[l], num=k_batch_size_list[l]).asnumpy()

                        if self.subtract_background:
                            _, std_l = estimate_bg(imgs_noise_l, bg_radius=self.bgRadius)
                            radial_psd_im = estimate_radial_psd(imgs_noise_l / std_l, basis, self.bgRadius)
                        else:
                            radial_psd_im = estimate_radial_psd(imgs_noise_l, basis, self.bgRadius, self.dtype)
                        if self.single_filter:
                            whiten_radial_full += weights * 1 / np.sqrt(np.mean(radial_psd_im, axis=0)).reshape(1, -1)
                        elif self.single_pass:
                            whiten_radial_full[k, :] += weights_k * 1 / np.sqrt(np.mean(radial_psd_im, axis=0))

                if self.single_filter or self.single_pass:
                    self.estimate_noise = False

        if store_images:
            if return_error:
                imgs_gt = np.zeros((num_denoise_imgs, img_size, img_size))
                imgs_raw = np.zeros((num_denoise_imgs, img_size, img_size))
            imgs_est = np.zeros((num_denoise_imgs, img_size, img_size))

        img_idx = 0
        img_idx_list = [0]
        for k in range(denoise_df_ct):
            k_batch_start, k_batch_size_list, k_batch_num = get_batch_idx(denoise_idx_num[k], self.batch_size)
            k_batch_start += denoise_idx_start[k]
            ctf_idx = denoise_df_id[k]

            for l in range(k_batch_num):
                if return_error:
                    imgs_clean_l = self.src.projections(start=k_batch_start[l], num=k_batch_size_list[l]).asnumpy()

                if self.flip_sign:
                    imgs_noise_l = -self.src.images(start=k_batch_start[l], num=k_batch_size_list[l]).asnumpy()
                else:
                    imgs_noise_l = self.src.images(start=k_batch_start[l], num=k_batch_size_list[l]).asnumpy()
                h_idx_l = np.zeros(k_batch_size_list[l]).astype(int)

                if self.subtract_background:
                    mean_l, std_l = estimate_bg(imgs_noise_l)
                    coeffs_eig_l = basis.evaluate_t((imgs_noise_l - mean_l) / std_l)

                else:
                    coeffs_eig_l = basis.evaluate_t(imgs_noise_l)


                coeffs_l = basis.to_angular_order(coeffs_eig_l.T).T
                coeffs_l = coeffs_l.reshape(k_batch_size_list[l], basis.ne)

                voltage_list = np.array([h_ctf[ctf_idx].voltage])
                cs_list = np.array([h_ctf[ctf_idx].Cs])
                alpha_list = np.array([h_ctf[ctf_idx].alpha])
                defocus_list = np.array([h_ctf[ctf_idx].defocus_mean])
                rwts_mat_l = basis.expand_ctf(voltage_list, cs_list, alpha_list, defocus_list, self.pixel_size)
                rwts_mat_l = basis.to_angular_order(rwts_mat_l.T).T

                if self.whiten:

                    if not self.estimate_noise:

                        if whiten_radial_full.shape[0] == 1:

                            whiten_radial = whiten_radial_full.reshape(1, -1)
                        elif whiten_radial_full.shape[0] == len(h_ctf) and self.noise_psd is not None:

                            whiten_radial = whiten_radial_full[ctf_idx, :].reshape(1, -1)
                        elif whiten_radial_full.shape[0] == denoise_df_ct and self.noise_psd is None:

                            whiten_radial = whiten_radial_full[k, :].reshape(1, -1)

                    else:

                        if self.subtract_background:
                            _, std_l = estimate_bg(imgs_noise_l, bg_radius=self.bgRadius)
                            radial_psd_im = estimate_radial_psd(imgs_noise_l / std_l, basis, self.bgRadius)
                        else:
                            radial_psd_im = estimate_radial_psd(imgs_noise_l, basis, self.bgRadius, self.dtype)

                        whiten_radial = 1 / np.sqrt(np.mean(radial_psd_im, axis=0)).reshape(1, -1)

                    whiten_fb_eig = basis.expand_raidal_vec(whiten_radial)
                    whiten_fb = basis.to_angular_order(whiten_fb_eig.T).T
                    coeffs_l = whiten_fb * coeffs_l
                    rwts_mat_l = whiten_fb * rwts_mat_l

                td0 = time.time()

                coeffs_est_l = wiener_filter(coeffs_l, mean_est, covar_est, self.noise_var, rwts_mat_l, h_idx_l, basis)

                td1 = time.time()
                self.t_denoise += td1 - td0
                if self.correct_contrast:
                    self.contrast_est[img_idx: img_idx + k_batch_size_list[l]] = coeffs_est_l[:,
                                                                                 basis.blk_ind[0]: basis.blk_ind[
                                                                                     1]] @ coeff_one0

                coeffs_est_l = basis.to_eigen_order(coeffs_est_l.T).T
                imgs_est_l = basis.evaluate(coeffs_est_l)


                if return_error:
                    err_denoise_l = LA.norm(imgs_clean_l - imgs_est_l, axis=(1, 2)) / LA.norm(imgs_clean_l, axis=(1, 2))
                    err_denoise_vec[img_idx: img_idx + k_batch_size_list[l]] = err_denoise_l

                if store_images:
                    imgs_est[img_idx: img_idx + k_batch_size_list[l]] = imgs_est_l

                    if return_error:
                        imgs_gt[img_idx: img_idx + k_batch_size_list[l]] = imgs_clean_l
                        imgs_raw[img_idx: img_idx + k_batch_size_list[l]] = imgs_noise_l
                img_idx = img_idx + k_batch_size_list[l]
            img_idx_list.append(img_idx)


        denoise_out = {
            "denoised_images": None,
            "clean_images": None,
            "mean_denoise_error": None,
            "denoise_error_vector": None,
            "image_indices_list": img_idx_list,
            "image_indices_start": denoise_idx_start,
            "image_indices_number": denoise_idx_num,
        }

        if store_images:
            denoise_out["denoised_images"] = imgs_est
            if return_error:
                denoise_out["clean_images"] = imgs_gt
                denoise_out["raw_images"] = imgs_raw

        if return_error:
            denoise_out["denoise_error_vector"] = err_denoise_vec
            denoise_out["mean_denoise_error"] = np.mean(err_denoise_vec)
        return denoise_out

    def estimate_mean(self):
        self.mean_est = np.zeros((self.basis.ne,))
        basis = self.basis
        mean_num = 0
        mean_deno = 0
        partition = []
        num_imgs = self.src.n
        h_ctf = self.src.unique_filters
        h_idx = self.src.filter_indices
        n_blk = basis.n_blk
        blk_size = basis.blk_size
        blk_ind = basis.blk_ind

        _, unique_count_all = np.unique(h_idx, return_counts=True)
        batch_start, batch_size_list, batch_num = get_batch_idx(num_imgs, self.batch_size)
        for ell in range(n_blk):
            partition.append([blk_size[ell], blk_size[ell]])

        self.covar_est = BlkDiagMatrix.zeros(partition, dtype=self.dtype)

        if self.whiten:
            tw0 = time.time()
            if self.single_filter:
                if self.estimate_noise:
                    self.noise_psd = np.zeros((1, basis.pts.shape[0]))
                    for l in range(0, batch_num):
                        weights = batch_size_list[l] / num_imgs

                        if self.flip_sign:
                            imgs_noise_l = -self.src.images(start=batch_start[l], num=batch_size_list[l]).asnumpy()
                        else:
                            imgs_noise_l = self.src.images(start=batch_start[l], num=batch_size_list[l]).asnumpy()

                        if self.subtract_background:
                            _, std_l = estimate_bg(imgs_noise_l, bg_radius=self.bgRadius)
                            radial_psd_im = estimate_radial_psd(imgs_noise_l / std_l, basis, self.bgRadius)
                        else:
                            radial_psd_im = estimate_radial_psd(imgs_noise_l, basis, self.bgRadius, self.dtype)

                        self.noise_psd += weights * np.mean(radial_psd_im, axis=0)
                else:
                    self.noise_psd = np.mean(self.noise_psd, axis=0).reshape(1, -1)

                self.estimate_noise = False

            elif self.store_psd and self.estimate_noise:

                self.noise_psd = np.zeros((len(h_ctf), basis.pts.shape[0]))
                for l in range(0, batch_num):

                    if self.flip_sign:
                        imgs_noise_l = -self.src.images(start=batch_start[l], num=batch_size_list[l]).asnumpy()
                    else:
                        imgs_noise_l = self.src.images(start=batch_start[l], num=batch_size_list[l]).asnumpy()
                    h_idx_l = h_idx[batch_start[l]:batch_start[l + 1]]
                    unique_val, unique_ind, unique_count = np.unique(h_idx_l, return_inverse=True,
                                                                     return_counts=True)

                    if self.subtract_background:
                        _, std_l = estimate_bg(imgs_noise_l, bg_radius=self.bgRadius)
                        radial_psd_im = estimate_radial_psd(imgs_noise_l / std_l, basis, self.bgRadius)
                    else:
                        radial_psd_im = estimate_radial_psd(imgs_noise_l, basis, self.bgRadius, self.dtype)

                    # radial_psd_df = np.zeros((len(unique_val), basis.pts.shape[0]))
                    for k_ind in range(len(unique_val)):
                        k = unique_val[k_ind]
                        weights = unique_count[k_ind] / unique_count_all[k]
                        self.noise_psd[k, :] += weights * np.mean(radial_psd_im[h_idx_l == k, :], axis=0)
                self.estimate_noise = False

            if self.noise_psd is not None:
                self.estimate_noise = False

            tw1 = time.time()
            self.t_whiten += tw1 - tw0

        for l in range(0, batch_num):
            weights = batch_size_list[l] / num_imgs

            if self.flip_sign:
                imgs_noise_l = -self.src.images(start=batch_start[l], num=batch_size_list[l]).asnumpy()
            else:
                imgs_noise_l = self.src.images(start=batch_start[l], num=batch_size_list[l]).asnumpy()

            # tbe0 = time.time()

            if self.subtract_background:
                mean_l, std_l = estimate_bg(imgs_noise_l)
                coeffs_eig_l = basis.evaluate_t((imgs_noise_l - mean_l) / std_l)

            else:
                coeffs_eig_l = basis.evaluate_t(imgs_noise_l)

            # tbe1 = time.time()
            # self.t_basis_expansion += tbe1 - tbe0
            coeffs_l0 = (basis.to_angular_order(coeffs_eig_l.T))[blk_ind[0]:blk_ind[1]].T
            h_idx_l = h_idx[batch_start[l]:batch_start[l + 1]]
            unique_val, unique_ind, unique_count = np.unique(h_idx_l, return_inverse=True, return_counts=True)

            voltage_list = np.array([h_ctf[k].voltage for k in unique_val])
            cs_list = np.array([h_ctf[k].Cs for k in unique_val])
            alpha_list = np.array([h_ctf[k].alpha for k in unique_val])
            defocus_list = np.array([h_ctf[k].defocus_mean for k in unique_val])
            tce0 = time.time()

            rwts_mat_l = basis.expand_ctf(voltage_list, cs_list, alpha_list, defocus_list, self.pixel_size)

            tce1 = time.time()
            self.t_ctf_expansion += tce1 - tce0
            rwts_mat_l0 = basis.to_angular_order(rwts_mat_l.T)[blk_ind[0]:blk_ind[1]].T

            if self.whiten:

                tw0 = time.time()

                if self.estimate_noise:
                    radial_psd_im = estimate_radial_psd(imgs_noise_l, basis, self.bgRadius, self.dtype)
                    radial_psd_df = np.zeros((len(unique_val), basis.pts.shape[0]))
                    for k_ind in range(len(unique_val)):
                        k = unique_val[k_ind]
                        radial_psd_df[k_ind, :] = np.mean(radial_psd_im[h_idx_l == k, :], axis=0)

                elif self.noise_psd.shape[0] == 1:
                    radial_psd_df = self.noise_psd
                else:
                    radial_psd_df = self.noise_psd[unique_val, :]

                # whiten_radial = np.zeros((num_whiten, basis.pts.shape[0]))
                whiten_radial = 1 / np.sqrt(radial_psd_df)

                whiten_fb_eig = basis.expand_raidal_vec(whiten_radial)
                whiten_fb0 = (basis.to_angular_order(whiten_fb_eig.T))[blk_ind[0]:blk_ind[1]].T

                rwts_mat_l0 = whiten_fb0 * rwts_mat_l0
                if whiten_fb0.shape[0] == 1:
                    coeffs_l0 = whiten_fb0 * coeffs_l0
                else:
                    coeffs_l0 = whiten_fb0[unique_ind, :] * coeffs_l0

                tw1 = time.time()
                self.t_whiten += tw1 - tw0
            coeffs_ctf_l0 = rwts_mat_l0[unique_ind, :] * coeffs_l0
            # mean estimation
            mean_num += weights * np.mean(coeffs_ctf_l0, 0)
            weights_vec = (unique_count / batch_size_list[l]).reshape(-1, 1)
            rwts_mat_l0_weighted = np.sqrt(weights_vec) * rwts_mat_l0
            mean_deno += weights * np.sum(rwts_mat_l0_weighted ** 2, 0)

        self.mean_est[blk_ind[0]:blk_ind[1]] = mean_num / mean_deno

        self.options["noise_psd"] = self.noise_psd
        if self.noise_psd is None:
            self.estimate_noise = True
        else:
            self.estimate_noise = False

        return self.mean_est

    def contrast_correction(self, mean_est=None, covar_est=None):

        if self.verbose:
            logger.info(f"correcting contrast for covariance estimation...")

        if mean_est is None or covar_est is None:
            if self.mean_est is None or self.covar_est is None:
                mean_est, covar_est = self.estimate_mean_covar()
            else:
                mean_est = self.mean_est
                covar_est = self.covar_est
        mean_R = self.basis.evaluate(self.basis.to_eigen_order(mean_est))
        im_one = np.ones((self.src.L, self.src.L))
        var_deno = np.sum(np.square(mean_R)) * np.sum(mean_R)
        coeff_one = self.basis.to_angular_order(self.basis.evaluate_t(im_one)).flatten()

        var_num0 = covar_est.apply(coeff_one)
        var_num = np.dot(mean_est, var_num0)
        var_est = var_num / var_deno
        blk_ind = self.basis.blk_ind
        mean_coeff0 = mean_est[blk_ind[0]: blk_ind[1]]
        mean_coeff_mat0 = mean_coeff0[:, np.newaxis]
        cov_mean0 = var_est * mean_coeff_mat0 @ mean_coeff_mat0.transpose()
        covar_est0 = covar_est[0]

        covar_X0 = (covar_est0 - cov_mean0) / (var_est + 1)
        covar_X0 = make_psd(covar_X0)

        coeff_one0 = coeff_one[blk_ind[0]:blk_ind[1]]

        Vmat_ell, Sval_ell, Vmat_ell_t = LA.svd(covar_X0)
        Vmat_one_ell = np.hstack((coeff_one0.reshape(-1, 1), Vmat_ell))
        Umat_ell_temp, _ = LA.qr(Vmat_one_ell[:, 0:-1])
        Umat_ell = np.hstack((Umat_ell_temp[:, 1:], Vmat_ell[:, -1].reshape(-1, 1)))
        covar_X0 = Umat_ell @ (np.diag(Sval_ell) @ Umat_ell.T)
        covar_X = covar_est.copy()
        covar_X[0] = covar_X0
        covar_est[0] = covar_X0 * (var_est + 1) + cov_mean0

        self.covar_est = covar_est
        self.covar_X = covar_X
        self.contrast_var = var_est

        return covar_est

