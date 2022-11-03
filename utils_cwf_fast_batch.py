import numpy as np
from aspire.volume import Volume
import scipy.sparse as spr
from scipy.linalg import solve, sqrtm
from aspire.utils import make_symmat, make_psd
from numpy.linalg import eig, inv
import scipy.linalg as LA
from aspire.operators import BlkDiagMatrix
from aspire.numeric import fft, xp
from aspire.utils.coor_trans import grid_2d
from aspire.source.simulation import Simulation
import finufft

def get_batch_idx(num_imgs, batch_size):
    batch_size_last = num_imgs % batch_size

    # split the image data into a number batches (at some point we need to handle large data)
    # to avoid using batched version, simply choose batch_size >= num_imgs, like in this example
    if batch_size_last != 0:
        batch_num = num_imgs // batch_size + 1
        batch_num = int(batch_num)
        batch_size_list = np.zeros((batch_num,))
        batch_size_list[0:batch_num - 1] = batch_size
        batch_size_list[-1] = batch_size_last
        batch_start = np.concatenate((np.zeros(1), np.cumsum(batch_size_list)))

    else:
        batch_num = num_imgs // batch_size
        batch_num = int(batch_num)
        batch_size_list = np.zeros((batch_num,))
        batch_size_list[:] = batch_size
        batch_start = np.concatenate((np.zeros(1), np.cumsum(batch_size_list)))

    batch_start = batch_start.astype(int)
    batch_size_list = batch_size_list.astype(int)


    return batch_start, batch_size_list, batch_num



def create_ordered_filter_idx(num_imgs, defocus_ct):

    df_size_last = num_imgs % defocus_ct

    if df_size_last != 0:
        df_size = num_imgs // (defocus_ct - 1)
        df_size_last = num_imgs % (defocus_ct - 1)

        h_idx = np.zeros((num_imgs,))
        h_idx[:num_imgs - df_size_last] = np.kron(range(0, defocus_ct - 1), np.ones((df_size,)))
        h_idx[num_imgs - df_size_last:] = defocus_ct - 1
        h_idx = h_idx.astype(int)
    else:
        df_size = num_imgs // defocus_ct
        h_idx = np.zeros((num_imgs,))
        h_idx = np.kron(range(0, defocus_ct), np.ones((df_size,)))
        h_idx = h_idx.astype(int)

    return h_idx


def mask_volume(vols, img_size, radius):
    imX = np.linspace(0, img_size - 1, img_size)
    imX = np.abs(imX - (img_size - 1) / 2.0)
    imX = np.square(imX)
    imX_full = np.tile(imX, (img_size, img_size, 1))
    imY_full = np.moveaxis(imX_full, 0, -1)
    imZ_full = np.moveaxis(imX_full, -1, 0)
    imXYZ = imX_full + imY_full + imZ_full
    imXYZ = np.sqrt(imXYZ)
    vol_mask = imXYZ <= radius
    vols = Volume(vol_mask * vols)

    return vols



def get_noise_var_batch(source, sn_ratio, batch_size):

    img_size = source.L
    num_imgs = source.n
    
    batch_start, batch_size_list, batch_num = get_batch_idx(num_imgs, batch_size)


    power_clean = 0

    for k in range(0, batch_num):
        imgs_ctf_clean_k = source.images(start=batch_start[k], num=batch_size_list[k])
        power_clean += imgs_ctf_clean_k.norm() ** 2

    power_clean = power_clean / (num_imgs * (img_size ** 2))

    noise_var = power_clean / sn_ratio

    return noise_var




def get_clean_mean_batch(source, basis, batch_size):
    print("computing the clean sample mean ...")
    num_imgs = source.n
    blk_ind = basis.blk_ind
    batch_start, batch_size_list, batch_num = get_batch_idx(num_imgs, batch_size)
    mean_clean = np.zeros((basis.ne,))
    for l in range(0, batch_num):

        print(f'drawing {l}-th batch of images')
        weights = batch_size_list[l] / num_imgs
        imgs_clean_l = source.projections(start=batch_start[l], num=batch_size_list[l]).asnumpy()
        coeffs_clean_eig_l = basis.evaluate_t(imgs_clean_l).reshape(batch_size_list[l], -1)
        coeffs_clean_l = basis.to_angular_order(coeffs_clean_eig_l.T).T
        mean_clean[blk_ind[0]:blk_ind[1]] += weights * np.mean(coeffs_clean_l[:, blk_ind[0]:blk_ind[1]], axis=0)
        
    return mean_clean     


def get_clean_covar_batch(source, basis, mean_clean, batch_size, dtype):
    print("computing the clean sample covariance ...")
    mean_clean = mean_clean.astype(dtype)
    num_imgs = source.n
    n_blk = basis.n_blk
    blk_size = basis.blk_size
    batch_start, batch_size_list, batch_num = get_batch_idx(num_imgs, batch_size)
    
    partition = []
    for ell in range(n_blk):
        partition.append([blk_size[ell], blk_size[ell]])

    covar_clean = BlkDiagMatrix.zeros(partition, dtype=dtype)

    for l in range(0, batch_num):
        print(f'drawing {l}-th batch of images')
        weights = batch_size_list[l] / num_imgs
        imgs_clean_l = source.projections(start=batch_start[l], num=batch_size_list[l]).asnumpy()
        coeffs_clean_eig_l = basis.evaluate_t(imgs_clean_l).reshape(batch_size_list[l], -1)
        coeffs_clean_l = basis.to_angular_order(coeffs_clean_eig_l.T).T
        covar_clean += weights * get_sample_covar(mean_clean, coeffs_clean_l, basis, dtype)
        
    return covar_clean    


def get_sample_covar(mean, coeffs, basis, dtype):

    mean = mean.astype(dtype)
    coeffs = coeffs.astype(dtype)
    blk_ind = basis.blk_ind

    covar_clean = BlkDiagMatrix.empty(0, dtype=dtype)
    if len(mean.shape) == 1:
        mean0 = mean[blk_ind[0]:blk_ind[1]]
    else:
        mean0 = mean[:, blk_ind[0]:blk_ind[1]]

    coeff_ell = coeffs[..., blk_ind[0]:blk_ind[1]] - mean0
    covar_ell = np.array(coeff_ell.T @ coeff_ell / coeffs.shape[0])
    covar_clean.append(covar_ell)

    # We'll also generate a mapping for complex construction
    indices_sgns = basis.indices_sgns
    indices_ells = basis.indices_ells
    for ell in range(1, basis.nmax + 1):
        mask = indices_ells == ell
        mask_pos = [
            mask[i] and (indices_sgns[i] == +1)
            for i in range(len(mask))
        ]
        mask_neg = [
            mask[i] and (indices_sgns[i] == -1)
            for i in range(len(mask))
        ]
        covar_ell_diag = np.array(
            coeffs[:, mask_pos].T @ coeffs[:, mask_pos]
            + coeffs[:, mask_neg].T @ coeffs[:, mask_neg]
        ) / (2 * coeffs.shape[0])

        covar_clean.append(covar_ell_diag)
        covar_clean.append(covar_ell_diag)

    return covar_clean



def shrink_covar(covar, noise_var, gamma, shrinker="frobenius_norm"):
    """
    Shrink the covariance matrix
    :param covar_in: An input covariance matrix
    :param noise_var: The estimated variance of noise
    :param gamma: An input parameter to specify the maximum values of eigen values to be neglected.
    :param shrinker: An input parameter to select different shrinking methods.
    :return: The shrinked covariance matrix
    """

    assert shrinker in (
        "frobenius_norm",
        "operator_norm",
        "soft_threshold",
    ), "Unsupported shrink method"

    lambs, eig_vec = eig(make_symmat(covar))

    lambda_max = noise_var * (1 + np.sqrt(gamma)) ** 2

    lambs[lambs < lambda_max] = 0

    if shrinker == "operator_norm":
        lambdas = lambs[lambs > lambda_max]
        lambdas = (
            1
            / 2
            * (
                lambdas
                - noise_var * (gamma - 1)
                + np.sqrt(
                    (lambdas - noise_var * (gamma - 1)) ** 2 - 4 * noise_var * lambdas
                )
            )
            - noise_var
        )

        lambs[lambs > lambda_max] = lambdas
    elif shrinker == "frobenius_norm":
        lambdas = lambs[lambs > lambda_max]
        lambdas = (
            1
            / 2
            * (
                lambdas
                - noise_var * (gamma - 1)
                + np.sqrt(
                    (lambdas - noise_var * (gamma - 1)) ** 2 - 4 * noise_var * lambdas
                )
            )
            - noise_var
        )
        c = np.divide(
            (1 - np.divide(noise_var**2 * gamma, lambdas**2)),
            (1 + np.divide(noise_var * gamma, lambdas)),
        )
        lambdas = lambdas * c
        lambs[lambs > lambda_max] = lambdas
    else:
        # for the case of shrinker == 'soft_threshold'
        lambdas = lambs[lambs > lambda_max]
        lambs[lambs > lambda_max] = lambdas - lambda_max

    diag_lambs = np.zeros_like(covar)
    np.fill_diagonal(diag_lambs, lambs)

    shrinked_covar = eig_vec @ diag_lambs @ eig_vec.conj().T

    return shrinked_covar

def shrink_covar_backward(b, b_noise, n, noise_var, shrinker, dtype):
    """
    Apply the shrinking method to the 2D covariance of coefficients.

    :param b: An input coefficient covariance.
    :param b_noise: The noise covariance.
    :param noise_var: The estimated variance of noise.
    :param shrinker: The shrinking method.
    :return: The shrinked 2D covariance coefficients.
    """
    b_out = b
    for ell in range(0, len(b)):
        b_ell = b[ell]
        p = np.size(b_ell, 1)
        # scipy >= 1.6.0 will upcast the sqrtm result to doubles
        #  https://github.com/scipy/scipy/issues/14853
        S = sqrtm(b_noise[ell]).astype(dtype)
        # from Matlab b_ell = S \ b_ell /S
        b_ell = solve(S, b_ell) @ inv(S)
        b_ell = shrink_covar(b_ell, noise_var, p / n, shrinker)
        b_ell = S @ b_ell @ S
        b_out[ell] = b_ell
    return b_out

def wiener_filter(coeffs_noise, mean_est, covar_est, noise_var, rwts_mat, h_idx, basis):

    blk_ind = basis.blk_ind
    noise_covar_coeff = noise_var * BlkDiagMatrix.eye_like(covar_est)
    coeffs_est = np.zeros_like(coeffs_noise)
    # coeffs_centered = coeffs_noise - rwts_mat[h_idx] * mean_est
    for k in np.unique(h_idx[:]):
        coeff_k = coeffs_noise[h_idx == k]
        coeff_est_k = coeff_k - rwts_mat[k] * mean_est
        sig_covar_coeff = BlkDiagMatrix.zeros_like(covar_est)
        for ell in range(sig_covar_coeff.nblocks):
            r_k = rwts_mat[k, blk_ind[ell]:blk_ind[ell + 1]].reshape(-1, 1)
            sig_covar_coeff[ell] = (r_k @ r_k.T) * covar_est[ell]

        sig_noise_covar_coeff = sig_covar_coeff + noise_covar_coeff
        coeff_est_k = sig_noise_covar_coeff.solve(coeff_est_k.T).T
        coeff_est_k = rwts_mat[k] * coeff_est_k
        coeff_est_k = covar_est.apply(coeff_est_k.T).T
        coeff_est_k = coeff_est_k + mean_est
        coeffs_est[h_idx == k] = coeff_est_k
    return coeffs_est



def compute_covar_err(covar1, covar2):
    err_vec = []

    mean_error_num = 0
    mean_error_deno = 0
    # blk_size = basis.blk_size
    # sum_size_sq = np.sum(blk_size ** 2)
    for ell in range(covar1.nblocks):
        err_num_ell = LA.norm(covar1[ell]-covar2[ell])
        err_deno_ell = LA.norm(covar2[ell])
        err_ell = err_num_ell / err_deno_ell
        # size_ell_sq = basis.blk_size[ell] ** 2
        mean_error_num += err_num_ell ** 2
        mean_error_deno += err_deno_ell ** 2
        err_vec.append(err_ell)

    mean_error = np.sqrt(mean_error_num/mean_error_deno)
    return err_vec, mean_error

def compute_radius_mat(img_size):
    g2d = grid_2d(img_size)
    rad_mat = g2d["r"]
    return rad_mat


def estimate_bg_mean_std(imgs, bg_radius=0.9):

    L = imgs.shape[-1]
    grid = grid_2d(L)
    mask = grid["r"] > bg_radius

    imgs_masked = imgs * mask
    denominator = np.sum(mask)
    first_moment = np.sum(imgs_masked, axis=(1, 2)) / denominator
    second_moment = np.sum(imgs_masked ** 2, axis=(1, 2)) / denominator
    mean = first_moment.reshape(-1, 1, 1)
    variance = second_moment.reshape(-1, 1, 1) - mean ** 2
    std = np.sqrt(variance)

    return mean, std


# def estimate_noise_psd(imgs, bgRadius=0.9):
#     """
#     :return: The estimated noise variance of the images in the Source used to create this estimator.
#     TODO: How's this initial estimate of variance different from the 'estimate' method?
#     """
#     # Run estimate using saved parameters
#     L = imgs.shape[-1]
#     nk = imgs.shape[0]
#     g2d = grid_2d(L)
#     mask = g2d["r"] >= bgRadius
#
#     images_masked = imgs * mask
#
#     _denominator = nk * np.sum(mask)
#     mean_est = np.sum(images_masked) / _denominator
#     im_masked_f = xp.asnumpy(fft.centered_fft2(xp.asarray(images_masked)))
#     noise_psd_est = np.sum(np.abs(im_masked_f ** 2), axis=0) / _denominator
#
#     mid = L // 2
#     #noise_psd_est[mid, mid] -= mean_est ** 2
#
#     return noise_psd_est



def estimate_radial_psd(imgs, basis, bgRadius=0.9, dtype=None):
    """
    :return: The estimated noise variance of the images in the Source used to create this estimator.
    TODO: How's this initial estimate of variance different from the 'estimate' method?
    """
    # Run estimate using saved parameters

    if dtype is None:
        dtype = np.float64

    L = imgs.shape[-1]
    n_im = imgs.shape[0]
    g2d = grid_2d(L)
    mask = g2d["r"] >= bgRadius


    if dtype == np.float64:
        complex_dtype = np.complex128
    else:
        complex_dtype = np.complex64

    images_masked = (imgs * mask).astype(complex_dtype)

    if n_im == 1:
        images_masked = images_masked.reshape(L, L)

    nufft_type = 2
    plan2_radial = finufft.Plan(nufft_type, (L, L), n_trans=n_im, eps=basis.eps, dtype=complex_dtype)

    plan2_radial.setpts(basis.grid_x.astype(dtype), basis.grid_y.astype(dtype))

    psd_polar = plan2_radial.execute(images_masked)
    psd_polar = psd_polar.reshape(n_im, basis.n_radial, basis.n_angular // 2)
    psd_polar = np.abs(psd_polar ** 2) / np.sum(mask)
    psd_radial = np.mean(psd_polar, axis=2)

    return psd_radial


def estimate_bg(imgs, bg_radius=0.9):

    L = imgs.shape[-1]
    grid = grid_2d(L)
    mask = grid["r"] > bg_radius

    imgs_masked = imgs * mask
    denominator = np.sum(mask)
    first_moment = np.sum(imgs_masked, axis=(1, 2)) / denominator
    second_moment = np.sum(imgs_masked ** 2, axis=(1, 2)) / denominator
    bg_mean = first_moment.reshape(-1, 1, 1)
    variance = second_moment.reshape(-1, 1, 1) - bg_mean ** 2
    bg_std = np.sqrt(variance)

    return bg_mean, bg_std


def compute_frc(imgs1, imgs2, n_angular, eps, dtype=None):

    if dtype is None:
        dtype = np.float64

    if dtype == np.float64:
        complex_dtype = np.complex128
    else:
        complex_dtype = np.complex64

    L = imgs1.shape[-1]
    imgs1 = imgs1.reshape(-1, L, L).astype(complex_dtype)
    imgs2 = imgs2.reshape(-1, L, L).astype(complex_dtype)
    n_im = imgs1.shape[0]

    R = L // 2
    h = 1 / R
    phi = 2 * np.pi * np.arange(n_angular // 2) / n_angular
    x = np.cos(phi)
    x = x.reshape(1, -1)
    y = np.sin(phi)
    y = y.reshape(1, -1)
    pts = np.arange(0, int(np.floor(L*1.5))).reshape(-1, 1)
    x = x * pts * h
    y = y * pts * h
    x = x.flatten().astype(dtype)
    y = y.flatten().astype(dtype)

    nufft_type = 2
    plan2_radial = finufft.Plan(nufft_type, (L, L), n_trans=n_im, eps=eps, dtype=complex_dtype)

    plan2_radial.setpts(x, y)

    f1 = plan2_radial.execute(imgs1)
    f2 = plan2_radial.execute(imgs2)
    f1 = f1.reshape(n_im, len(pts), n_angular // 2)
    f2 = f2.reshape(n_im, len(pts), n_angular // 2)
    corr_num = np.real(np.sum(f1 * np.conj(f2), axis=-1))
    corr_deno1 = np.sqrt(np.abs(np.sum(f1 * np.conj(f1), axis=-1)))
    corr_deno2 = np.sqrt(np.abs(np.sum(f2 * np.conj(f2), axis=-1)))
    frc_mat = corr_num / (corr_deno1 * corr_deno2)
    frc_vec = np.mean(frc_mat, axis=0)

    return frc_mat, frc_vec


def compute_img_err(imgs1, imgs2, radius):

    L = imgs1.shape[-1]
    imgs1 = imgs1.reshape(-1, L, L)
    imgs2 = imgs2.reshape(-1, L, L)
    grid = grid_2d(L)
    mask = grid["r"] <= radius
    err_vec = LA.norm(imgs1[:, mask] - imgs2[:, mask], axis=-1) / LA.norm(imgs2[:, mask], axis=-1)
    mean_err = np.mean(err_vec)

    return err_vec, mean_err