import time

import numpy as np
import scipy
import scipy.special as spl
import scipy.sparse as spr
import finufft
from scipy.io import loadmat
from joblib import Parallel, delayed
import os


class FLEBasis2D:
    #
    #   L           basis for L x L images
    #   bandlimit   bandlimit parameter (scaled so that L is max suggested)
    #   eps         requested relative precision
    #   maxitr      maximum number of interations for expand method
    #   maxfun      maximum number of basis functions to use
    #   mode        choose either "real" or "complex" output
    #
    def __init__(
            self,
            L,
            bandlimit,
            eps,
            maxitr=None,
            maxfun=None,
            mode="real",
            dtype=np.float32
    ):

        realmode = mode == "real"
        complexmode = mode == "complex"
        assert realmode or complexmode

        self.complexmode = complexmode
        self.dtype = dtype
        if dtype == np.float64:
            self.complex_dtype = np.complex128
        else:
            self.complex_dtype = np.complex64

        # Heuristics for choosing numsparse and maxitr
        maxitr = 1 + int(3 * np.log2(L))
        numsparse = 32
        if eps >= 1e-10:
            numsparse = 22
            maxitr = 1 + int(2 * np.log2(L))
        if eps >= 1e-7:
            numsparse = 16
            maxitr = 1 + int(np.log2(L))
        if eps >= 1e-4:
            numsparse = 8
            maxitr = 1 + int(np.log2(L)) // 2

        self.W = self.precomp(
            L,
            bandlimit,
            eps,
            maxitr,
            numsparse,
        )

    def precomp(
            self,
            L,
            bandlimit,
            eps,
            maxitr,
            numsparse,
            maxfun=None,
    ):

        # see {eq:num_radial_nodes}
        # n_radial = int(max(2.1 * L, np.log2(1 / eps)))

        Q = int(np.ceil(2.4 * L))
        n_radial = Q
        tmp = 1 / (np.sqrt(np.pi))
        for q in range(1, Q + 1):
            tmp = tmp / q * (np.sqrt(np.pi) * L / 4)
            if tmp <= eps:
                n_radial = int(max(q, np.log2(1 / eps)))
                break
        n_radial = max(n_radial, int(np.ceil(np.log2(1 / eps))))

        # if n_radial % 2 == 0:
        #    n_radial += 1

        n_interp = n_radial
        # if numsparse > 0:
        #     n_interp = 2 * n_radial

        if maxfun:
            ne = maxfun
        else:
            ne = int(L ** 2 * np.pi / 4)

        max_bandlimit = L
        psi, ns, ks, lmds, cs, ne = self.lap_eig_disk(
            ne, bandlimit, max_bandlimit
        )

        nmax = np.max(np.abs(ns))
        b_sz = (n_interp, 2 * nmax + 1)
        b = np.zeros(b_sz, dtype=self.dtype)

        ndx = 2 * np.abs(ns) - (ns < 0)
        ndmax = np.max(ndx)
        idx_list = [None] * (ndmax + 1)
        for i in range(ndmax + 1):
            idx_list[i] = []
        for i in range(ne):
            nd = ndx[i]
            idx_list[nd].append(i)

        nmax = np.max(np.abs(ns))
        v = 0
        for i in range(len(idx_list)):
            v = max(len(idx_list[i]), v)

        # Source points
        xs = 1 - (2 * np.arange(n_interp) + 1) / (2 * n_interp)
        xs = np.cos(np.pi * xs)

        lmd0 = np.min(lmds)
        lmd1 = np.max(lmds)

        if ne == 1:
            if self.dtype == np.float64:
                lmd1 = lmd1 * (1 + 2e-16)
            else:
                lmd1 = lmd1 * (1 + 2e-8)

        a = np.zeros(ne, dtype=self.dtype)

        # Make a list of lists: idx_list[i] is the index of all values i
        ndx = 2 * np.abs(ns) - (ns < 0)
        ndmax = np.max(ndx)
        idx_list = [None] * (ndmax + 1)
        for i in range(ndmax + 1):
            idx_list[i] = []
        for i in range(ne):
            nd = ndx[i]
            idx_list[nd].append(i)

        nmax = np.max(np.abs(ns))
        v = 0
        for i in range(len(idx_list)):
            v = max(len(idx_list[i]), v)

        # Source points
        xs = 1 - (2 * np.arange(n_interp) + 1) / (2 * n_interp)
        xs = np.cos(np.pi * xs)

        if numsparse <= 0:
            ws = self.get_weights(xs)

        a = np.zeros(ne, dtype=self.dtype)
        A3 = [None] * (ndmax + 1)
        A3_T = [None] * (ndmax + 1)
        for i in range(ndmax + 1):

            # Source function values
            ys = b[:, i]
            ys = ys.flatten()

            # Target points
            x = 2 * (lmds[idx_list[i]] - lmd0) / (lmd1 - lmd0) - 1
            vals, x_ind, xs_ind = np.intersect1d(x, xs, return_indices=True)
            if self.dtype == np.float64:
                x[x_ind] = x[x_ind] + 2e-16
            else:
                x[x_ind] = x[x_ind] + 2e-8

            n = len(x)
            mm = len(xs)

            # if s is less than or equal to 0 we just to dense
            if numsparse > 0:
                A3[i], A3_T[i] = self.barycentric_interp_sparse(
                    x, xs, ys, numsparse
                )
            else:
                A3[i] = np.zeros((n, mm), dtype=self.dtype)
                numer = np.zeros(n, dtype=self.dtype)
                denom = np.zeros(n, dtype=self.dtype)
                exact = np.zeros(n, dtype=self.dtype)
                for j in range(mm):
                    xdiff = x - xs[j]
                    temp = ws[j] / xdiff
                    A3[i][:, j] = temp.flatten()
                    denom = denom + temp
                denom = denom.reshape(-1, 1)
                A3[i] = A3[i] / denom
                A3_T[i] = A3[i].T

        R = L // 2
        h = 1 / R
        x = np.arange(-R, R + L % 2)
        y = np.arange(-R, R + L % 2)
        xs, ys = np.meshgrid(x, y)
        xs = xs / R
        ys = ys / R
        rs = np.sqrt(xs ** 2 + ys ** 2)

        if self.dtype == np.float64:
            idx = rs > 1 + 1e-13
        else:
            idx = rs > 1 + 1e-6

        ndx = 2 * np.abs(ns) - (ns < 0)
        ndmax = np.max(ndx)

        # n_angular {eq:num_angular_nodes}
        # n_angular = max(2 * (lmd1 + nmax), np.log2(1 / eps))
        # n_angular = 2 * int(n_angular / 2)

        # ### Explicit calculation
        S = int(max(7.08 * L, -np.log2(eps) + 2 * np.log2(L)))
        n_angular = S
        for svar in range(
                int(lmd1 + ndmax) + 1, S + 1
        ):  # we used s somewhere else, so using svar here
            tmp = (
                    L ** 2 * ((lmd1 + ndmax) / svar) ** svar
            )  # ndmax = n_m from the writeup, right?
            if tmp <= eps:
                n_angular = int(max(int(svar), np.log2(1 / eps)))
                break

        # print(n_angular)
        if n_angular % 2 == 1:
            n_angular += 1

        c2r = self.precomp_transform_complex_to_real(ns)
        r2c = spr.csr_matrix(c2r.transpose().conj())

        # Fast nus transform
        nus = np.zeros(1 + 2 * nmax, dtype=int)
        nus[0] = 0
        for i in range(1, nmax + 1):
            nus[2 * i - 1] = -i
            nus[2 * i] = i
        c2r_nus = self.precomp_transform_complex_to_real(nus)
        r2c_nus = spr.csr_matrix(c2r_nus.transpose().conj())

        xs = 1 - (2 * np.arange(n_radial) + 1) / (2 * n_radial)
        xs = np.cos(np.pi * xs)
        pts = (xs + 1) / 2
        pts = (lmd1 - lmd0) * pts + lmd0
        pts = pts.reshape(-1, 1)

        blk_size = np.zeros(2 * nmax + 1)
        for i in range(2 * nmax + 1):
            blk_size[i] = len(idx_list[i])

        blk_size = blk_size.astype(int)
        k_max = blk_size[::2]
        blk_ind = np.concatenate((np.zeros(1), np.cumsum(blk_size)))
        blk_ind = blk_ind.astype(int)
        n_blk = len(blk_ind) - 1
        indices_sgns = np.zeros(ne, dtype=int)
        indices_ells = np.zeros(ne, dtype=int)
        i = 0
        ci = 0
        for ell in range(nmax + 1):
            sgns = (1,) if ell == 0 else (1, -1)
            ks = np.arange(0, k_max[ell])

            for sgn in sgns:
                rng = np.arange(i, i + len(ks))
                indices_ells[rng] = ell
                indices_sgns[rng] = sgn

                i += len(ks)

            ci += len(ks)

        ind_vec = (nus % 2 - 2) * -np.sign(nus)

        self.ind_vec = ind_vec
        self.indices_sgns = indices_sgns
        self.indices_ells = indices_ells
        self.pts = pts
        self.blk_ind = blk_ind
        self.k_max = k_max
        self.n_blk = n_blk
        self.blk_size = blk_size
        self.ns = ns
        self.ks = ks
        self.lmds = lmds
        self.L = L
        self.ne = ne
        self.cs = cs
        self.n_radial = n_radial
        self.n_interp = n_interp
        self.rs = rs
        self.A3 = A3
        self.A3_T = A3_T
        self.idx_list = idx_list
        self.ndmax = ndmax
        self.h = h
        self.maxitr = maxitr
        self.eps = eps
        self.idx = idx
        self.n_angular = n_angular
        self.nmax = nmax
        self.psi = psi
        self.c2r = c2r
        self.r2c = r2c
        self.c2r_nus = c2r_nus
        self.r2c_nus = r2c_nus
        self.nus = nus
        self.ndx = ndx
        self.lmd0 = lmd0
        self.lmd1 = lmd1

        if L < 16:
            self.B = self.create_denseB()

        R = L // 2
        h = 1 / R
        phi = 2 * np.pi * np.arange(self.n_angular // 2) / self.n_angular
        x = np.cos(phi)
        x = x.reshape(1, -1)
        y = np.sin(phi)
        y = y.reshape(1, -1)
        x = x * pts * h
        y = y * pts * h
        x = x.flatten().astype(self.dtype)
        y = y.flatten().astype(self.dtype)

        self.grid_x = x
        self.grid_y = y
        nufft_type = 2
        self.plan2 = finufft.Plan(nufft_type, (L, L), n_trans=1, eps=eps, dtype=self.complex_dtype)
        self.plan2.setpts(x, y)

        nufft_type = 1
        self.plan1 = finufft.Plan(nufft_type, (L, L), n_trans=1, eps=eps, dtype=self.complex_dtype)
        self.plan1.setpts(x, y)

        return

    def radialconv_wts(self, b):

        ne = self.ne

        b = np.array(b, order="F")

        h = self.h

        a = np.zeros(ne, dtype=self.dtype)

        y = [None] * (self.ndmax + 1)
        for i in range(self.ndmax + 1):
            y[i] = (self.A3[i] @ b[:, 0]).flatten()

        for i in range(self.ndmax + 1):
            a[self.idx_list[i]] = y[i]

        return a.flatten()

    def radialconv(self, a, f):

        a = a.flatten()
        b = self.radialconv_multipliers(f).flatten()
        if self.complexmode:
            a_conv = a * b
        else:
            a_conv = self.c2r @ (b * (self.r2c @ a).flatten())

        return a_conv.flatten()

    def to_angular_order(self, a):
        a_ordered = np.zeros_like(a)
        blk_ind = self.blk_ind
        idx_list = self.idx_list
        for i in range(len(blk_ind) - 1):
            idx_i = idx_list[i]
            a_ordered[blk_ind[i]:blk_ind[i + 1]] = a[idx_i]

        return a_ordered

    def to_eigen_order(self, a_ordered):
        a = np.zeros_like(a_ordered)
        blk_ind = self.blk_ind
        idx_list = self.idx_list
        for i in range(len(blk_ind) - 1):
            idx_i = idx_list[i]
            a[idx_i] = a_ordered[blk_ind[i]:blk_ind[i + 1]]

        return a

    def expand_ctf(self, voltage_list, cs_list, alpha_list, defocus_list, pixel_size):
        pts = self.pts
        h = self.h
        wavelength_list = 12.2643247 / np.sqrt(voltage_list * 1e3 + 0.978466 * voltage_list ** 2)
        c2_vec = (-np.pi * wavelength_list * defocus_list).reshape(-1, 1)
        c4_vec = (0.5 * np.pi * (cs_list * 1e7) * wavelength_list ** 3).reshape(-1, 1)

        r2 = (pts * h / (pixel_size * 2 * np.pi)) ** 2
        r4 = r2 ** 2
        gamma = r2 @ c2_vec.T + r4 @ c4_vec.T
        ctf_radial = np.sqrt(1 - alpha_list ** 2) * np.sin(gamma) - alpha_list * np.cos(gamma)

        rwts_mat = self.expand_raidal_vec(ctf_radial.T)

        return rwts_mat

    def expand_raidal_vec(self, radial_vec):

        radial_vec = radial_vec.T

        radial_fb = np.zeros((self.ne, radial_vec.shape[1]), dtype=self.dtype)

        for i in range(self.ndmax + 1):
            radial_fb[self.idx_list[i], :] = self.A3[i] @ radial_vec

        return radial_fb.T

    def rotate(self, a, theta):

        a = a.flatten()
        b = self.rotate_multipliers(theta).flatten()
        if self.complexmode:
            a_rot = a * b
        else:
            a_rot = self.c2r @ (b * (self.r2c @ a).flatten())

        return a_rot.flatten()

    def radialconv_multipliers(self, f):

        # Copy and reshape f
        L = self.L
        f = np.copy(f).reshape(L, L)

        # Remove pixels outside disk
        f[self.idx] = 0
        f = f.flatten()

        # Step 1.
        z = self.step1(f)

        # Step 2.
        b = self.step2(z)

        # Step 3.
        wts = self.radialconv_wts(b)

        b = wts / self.h ** 2

        ne = self.ne
        return b.reshape(ne)

    def evaluate_t(self, f):
        # see {sec:fast_details}

        L = self.L

        if np.prod(f.shape) == self.L ** 2:

            f = np.copy(f).reshape(self.L, self.L).astype(self.dtype)

            # For small images just use matrix multiplication
            if L < 16:
                return (self.B.T @ f.flatten()).flatten()

            # Remove pixels outside disk
            f[self.idx] = 0
            f = f.flatten()

        else:
            nf = f.shape[0]

            f = np.copy(f).reshape(nf, self.L, self.L)

            # For small images just use matrix multiplication
            if L < 16:
                return (self.B.T @ f.reshape(nf, L ** 2).T).T

            # Remove pixels outside disk

            # For small images just use matrix multiplication
            # Remove pixels outside disk
            f[:, self.idx] = 0
            f = f.reshape(nf, L ** 2)

        t10 = time.time()
        # Step 1. {sec:fast_details}
        z = self.step1(f)
        t11 = time.time()


        t20 = time.time()

        # Step 2. {sec:fast_details}
        b = self.step2(z)

        t21 = time.time()


        t30 = time.time()

        # Step 3: {sec:fast_details}
        a = self.step3(b)

        t31 = time.time()


        if self.complexmode:

            if np.prod(f.shape) == self.L ** 2:
                a = self.r2c @ a.flatten()
                a = a.reshape(self.ne)

            else:
                a = (self.r2c @ a.T).T

        return a

    def evaluate(self, a):
        # see {rmk:how_to_apply_B} and {sec:fast_details}

        L = self.L

        if np.prod(a.shape) == self.ne:

            if self.complexmode:
                a = np.real(self.c2r @ a.flatten())
                a = a.reshape(self.ne)

            a = np.real(a)
            # for small images use matrix multiplication
            if self.L < 16:
                return (self.B @ a).reshape(self.L, self.L)

        else:
            na = a.shape[0]
            if self.complexmode:
                a = (self.c2r @ a.T).T

            a = np.real(a)
            if self.L < 16:
                return (self.B @ a.T).reshape(na, self.L, self.L)

        # B1
        b = self.step3_H(a)

        # B2
        z = self.step2_H(b)

        # B3
        f = self.step1_H(z)

        if np.prod(a.shape) == self.ne:
            f = f.reshape(L, L)
        else:
            na = a.shape[0]
            f = f.reshape(na, L, L)

        return f

    def expand(self, f):

        b = self.evaluate_t(f)
        a0 = b
        for i in range(self.maxitr):
            a0 = a0 - self.evaluate_t(self.evaluate(a0)) + b
        return a0

    def step1(self, f):

        if np.prod(f.shape) == self.L ** 2:
            f = f.reshape(self.L, self.L)
            f = np.array(f, dtype=self.complex_dtype)
            z = np.zeros((self.n_radial, self.n_angular), dtype=self.complex_dtype)
            z0 = self.plan2.execute(f) * self.h ** 2
            z0 = z0.reshape(self.n_radial, self.n_angular // 2)
            z[:, : self.n_angular // 2] = z0
            z[:, self.n_angular // 2:] = np.conj(z0)
            z = z.flatten()
        else:

            L = self.L
            nf = f.shape[0]
            f = f.reshape(nf, L, L)
            f = np.array(f, dtype=self.complex_dtype)

            z = np.zeros((nf, self.n_radial, self.n_angular), dtype=self.complex_dtype)
            nufft_type = 2
            plan2v = finufft.Plan(nufft_type, (L, L), n_trans=nf, eps=self.eps, dtype=self.complex_dtype)
            plan2v.setpts(self.grid_x, self.grid_y)

            z0 = plan2v.execute(f) * self.h ** 2
            z0 = z0.reshape(nf, self.n_radial, self.n_angular // 2)

            z[:, :, : self.n_angular // 2] = z0
            z[:, :, self.n_angular // 2:] = np.conj(z0)
            z = z.reshape(nf, self.n_angular * self.n_radial)

        return z

    def step1_H(self, z):
        if np.prod(z.shape) == self.n_radial * self.n_angular:

            # Half z
            z = z[:, : self.n_angular // 2]
            f = self.plan1.execute(z.flatten())
            f = f + np.conj(f)
            f = np.real(f)
            f = f.reshape(self.L, self.L)
            f[self.idx] = 0

            f = f.flatten()

        else:

            nz = z.shape[0]
            z = z[:, :, : self.n_angular // 2]
            nufft_type = 1
            plan1v = finufft.Plan(nufft_type, (self.L, self.L), n_trans=nz, eps=self.eps, dtype=self.complex_dtype)
            plan1v.setpts(self.grid_x, self.grid_y)
            f = plan1v.execute(z.reshape(nz, -1))
            f = f + np.conj(f)
            f = np.real(f)
            f = f.reshape(nz, self.L, self.L)
            f[:, self.idx] = 0

        return f

    def step2(self, z):

        if np.prod(z.shape) == self.n_radial * self.n_angular:
            # Compute Fourier coefficients along rings
            z = z.reshape(self.n_radial, self.n_angular).T
            b = scipy.fft.fft(z, n=self.n_angular, axis=0) / self.n_angular
            b = b[self.nus, :]
            b = np.conj(b)
            b = self.c2r_nus @ b
            b = np.real(b).T

        else:

            nz = z.shape[0]
            z = z.reshape(nz, self.n_radial, self.n_angular)
            z = np.swapaxes(z, 0, 2) ###

            t_fft0 = time.time()

            # b_temp = scipy.fft.fft(z, n=self.n_angular, axis=2) / self.n_angular
            b = scipy.fft.fft(z, n=self.n_angular, axis=0) / self.n_angular ###

            t_fft1 = time.time()

            t_fft = t_fft1 - t_fft0


            t_sub0 = time.time()
            # b = b_temp[:, :, self.nus]
            b = b[self.nus, :, :]

            t_sub1 = time.time()
            t_sub = t_sub1 - t_sub0


            t_conj0 = time.time()
            b = np.conj(b)

            t_conj1 = time.time()
            t_conj = t_conj1 - t_conj0


            # b = np.swapaxes(b, 0, 2)
            b = b.reshape(-1, self.n_radial * nz)

            t_mult0 = time.time()
            b = self.c2r_nus @ b

            t_mult1 = time.time()
            t_mult = t_mult1 - t_mult0


            t_swap0 = time.time()

            b = b.reshape(-1, self.n_radial, nz)
            b = np.real(np.swapaxes(b, 0, 2))

            t_swap1 = time.time()
            t_swap = t_swap1 - t_swap0




        return b

    def step3(self, b):

        if len(b.shape) == 2:

            b = np.array(b, order="F")

            h = self.h

            a = np.zeros(self.ne, dtype=self.dtype)
            y = [None] * (self.ndmax + 1)

            for i in range(self.ndmax + 1):
                y[i] = (self.A3[i] @ b[:, i]).flatten()

            for i in range(self.ndmax + 1):
                a[self.idx_list[i]] = y[i]

            a = a * self.cs / h
            a = a.flatten()


        else:

            nb = b.shape[0]

            h = self.h

            b = np.moveaxis(b, 0, -1)

            t_intp0 = time.time()

            a = np.zeros((self.ne, nb), dtype=self.dtype)

            for i in range(self.ndmax + 1):
                a[self.idx_list[i]] = self.A3[i] @ b[:, i, :]

            t_intp1 = time.time()
            t_intp = t_intp1 - t_intp0


            a = a.T

            a = a * self.cs / h

        return a

    def step3_H(self, a):

        if np.prod(a.shape) == self.ne:

            h = self.h

            a = a * h
            a = a.flatten()
            a = a * self.cs

            y = [None] * (self.ndmax + 1)
            for i in range(self.ndmax + 1):
                y[i] = a[self.idx_list[i]]

            b = np.zeros(
                (self.n_interp, 2 * self.nmax + 1), dtype=self.dtype, order="F"
            )
            for i in range(self.ndmax + 1):
                b[:, i] = self.A3_T[i] @ y[i]


        else:

            na = a.shape[0]

            a = a.reshape(na, self.ne)
            # for small images use matrix multiplication

            h = self.h

            a = a * h * self.cs

            a = a.T
            b = np.zeros(
                (self.n_interp, 2 * self.nmax + 1, na), dtype=self.dtype, order="F"
            )
            for i in range(self.ndmax + 1):
                b[:, i, :] = self.A3_T[i] @ a[self.idx_list[i]]

            b = np.moveaxis(b, -1, 0)

        return b

    def step2_H(self, b):

        if len(b.shape) == 2:

            tmp = np.zeros((b.shape[0], self.n_angular), dtype=self.complex_dtype)
            tmp0 = (self.r2c_nus @ b.T).T
            tmp[:, self.nus] = np.conj(tmp0)
            z = scipy.fft.ifft(tmp, axis=1)

        else:

            nb = b.shape[0]
            tmp = np.zeros((b.shape[0], b.shape[1], self.n_angular), dtype=self.complex_dtype)

            b = np.swapaxes(b, 0, 2)
            b = b.reshape(-1, self.n_radial * nb)
            b = self.r2c_nus @ b
            b = b.reshape(-1, self.n_radial, nb)
            b = np.swapaxes(b, 0, 2)

            tmp[:, :, self.nus] = np.conj(b)
            z = scipy.fft.ifft(tmp, axis=2)

        return z

    def create_denseB(self, numthread=1):
        # see {eq:operator_B} and {eq:operator_B^*}

        # Evaluate eigenfunctions
        R = self.L // 2
        h = 1 / R
        x = np.arange(-R, R + self.L % 2)
        y = np.arange(-R, R + self.L % 2)
        xs, ys = np.meshgrid(x, y)
        xs = xs / R
        ys = ys / R
        rs = np.sqrt(xs ** 2 + ys ** 2)
        ts = np.arctan2(ys, xs)

        # Compute in parallel if numthread > 1
        if numthread <= 1:
            B = np.zeros(
                (self.L, self.L, self.ne), dtype=self.complex_dtype, order="F"
            )
            for i in range(self.ne):
                B[:, :, i] = self.psi[i](rs, ts)
            B = h * B
        else:
            func = lambda i, rs=rs, ts=ts: self.psi[i](rs, ts)
            B_list = Parallel(n_jobs=numthread, prefer="threads")(
                delayed(func)(i) for i in range(self.ne)
            )
            B_par = np.zeros(
                (self.L, self.L, self.ne), dtype=self.complex_dtype, order="F"
            )
            for i in range(self.ne):
                B_par[:, :, i] = B_list[i]
            B = h * B_par

        B = B.reshape(self.L ** 2, self.ne)
        B = self.transform_complex_to_real(np.conj(B), self.ns)

        return B.reshape(self.L ** 2, self.ne)

    def lap_eig_disk(self, ne, bandlimit, max_bandlimit):

        # number of roots to check
        nc = int(3 * np.sqrt(ne))
        nd = int(2 * np.sqrt(ne))

        # preallocate
        nn = 1 + 2 * nc
        ns = np.zeros((nn, nd), dtype=int, order="F")
        ks = np.zeros((nn, nd), dtype=int, order="F")
        lmds = np.ones((nn, nd), dtype=self.dtype) * np.Inf

        # load table of roots of jn (the scipy code has an issue where it gets
        # stuck in an infinite loop in Newton's method as of Jun 2022)
        path_to_module = os.path.dirname(__file__)
        zeros_path = os.path.join(path_to_module, "jn_zeros_n=3000_nt=2500.mat")
        data = loadmat(zeros_path)
        roots_table = data["roots_table"]

        ns[0, :] = 0
        lmds[0, :] = roots_table[0, :nd]
        ks[0, :] = np.arange(nd) + 1

        # add roots of J_n for n > 0 twice with +k and -k
        # the square of the roots are eigenvalues of the Laplacian (with
        # Dirichlet boundary conditions
        # see {eq:eigenfun}
        for i in range(nc):
            n = i + 1
            ns[2 * n - 1, :] = -n
            ks[2 * n - 1, :] = np.arange(nd) + 1

            lmds[2 * n - 1, :nd] = roots_table[n, :nd]

            ns[2 * n, :] = n
            ks[2 * n, :] = ks[2 * n - 1, :]
            lmds[2 * n, :] = lmds[2 * n - 1, :]

        # flatten
        ns = ns.flatten()
        ks = ks.flatten()
        lmds = lmds.flatten()

        # sort by lmds
        idx = np.argsort(lmds)
        ns = ns[idx]
        ks = ks[idx]
        lmds = lmds[idx]

        # sort complex conjugate pairs: -n first, +n second
        idx = np.arange(ne + 1)

        if self.dtype == np.float64:
            for i in range(ne + 1):
                if ns[i] >= 0:
                    continue
                if np.abs(lmds[i] - lmds[i + 1]) < 1e-14:
                    continue
                idx[i - 1] = i
                idx[i] = i - 1
        else:
            for i in range(ne + 1):
                if ns[i] >= 0:
                    continue
                if np.abs(lmds[i] - lmds[i + 1]) < 1e-7:
                    continue
                idx[i - 1] = i
                idx[i] = i - 1

        ns = ns[idx]
        ks = ks[idx]
        lmds = lmds[idx]

        # {sec:bandlimit}
        if bandlimit:
            for i in range(len(lmds)):
                if lmds[ne] / (np.pi) >= (bandlimit - 1) // 2:
                    ne = ne - 1

        # potentially subjtract 1 from ne to keep complex conj pairs
        if ns[ne - 1] < 0:
            ne = ne - 1

        # make sure that ne is always at least 1
        if ne <= 1:
            ne = 1

        # take top ne values
        ns = ns[:ne]
        ks = ks[:ne]
        lmds = lmds[:ne]

        cs = np.zeros(ne, dtype=self.dtype)

        psi = [None] * ne
        for i in range(ne):
            n = ns[i]
            lmd = lmds[i]
            # see {eq:eigenfun_const}
            c = 1 / np.sqrt(0.5 * np.pi * spl.jv(ns[i] + 1, lmds[i]) ** 2)
            if ns[i] == 0:
                c /= np.sqrt(2)
                # see {eq:eigenfun} and {eq:eigenfun_extend}
                psi[i] = (
                    lambda r, t, n=n, c=c, lmd=lmd: c
                                                    * spl.jv(n, lmd * r)
                                                    * (r <= 1)
                )
            else:
                # see {eq:eigenfun_const}
                c = c / np.sqrt(2)
                # see {eq:eigenfun} and {eq:eigenfun_extend}
                psi[i] = (
                    lambda r, t, c=c, n=n, lmd=lmd: c
                                                    * spl.jv(n, lmd * r)
                                                    * np.exp(1j * n * t)
                                                    * (r <= 1)
                                                    * (-1) ** np.abs(n)
                )
            cs[i] = c

        return psi, ns, ks, lmds, cs, ne

    def precomp_transform_complex_to_real(self, ns):

        ne = len(ns)
        nnz = np.sum(ns == 0) + 2 * np.sum(ns != 0)
        idx = np.zeros(nnz, dtype=int)
        jdx = np.zeros(nnz, dtype=int)
        vals = np.zeros(nnz, dtype=self.complex_dtype)

        k = 0
        for i in range(ne):
            n = ns[i]
            if n == 0:
                vals[k] = 1
                idx[k] = i
                jdx[k] = i
                k = k + 1
            if n < 0:
                s = (-1) ** np.abs(n)

                vals[k] = 1 / np.sqrt(2)
                idx[k] = i
                jdx[k] = i
                k = k + 1

                vals[k] = s / np.sqrt(2)
                idx[k] = i
                jdx[k] = i + 1
                k = k + 1

                vals[k] = -1 / (1j * np.sqrt(2))
                idx[k] = i + 1
                jdx[k] = i
                k = k + 1

                vals[k] = s / (1j * np.sqrt(2))
                idx[k] = i + 1
                jdx[k] = i + 1
                k = k + 1

        A = spr.csr_matrix(
            (vals, (idx, jdx)), shape=(ne, ne), dtype=self.complex_dtype
        )

        return A

    def transform_complex_to_real(self, Z, ns):

        ne = Z.shape[1]
        X = np.zeros(Z.shape, dtype=self.dtype)

        for i in range(ne):
            n = ns[i]
            if n == 0:
                X[:, i] = np.real(Z[:, i])
            if n < 0:
                s = (-1) ** np.abs(n)
                x0 = (Z[:, i] + s * Z[:, i + 1]) / np.sqrt(2)
                x1 = (-Z[:, i] + s * Z[:, i + 1]) / (1j * np.sqrt(2))
                X[:, i] = np.real(x0)
                X[:, i + 1] = np.real(x1)

        return X

    def transform_real_to_complex(self, X, ns):

        ne = X.shape[1]
        Z = np.zeros(X.shape, dtype=self.complex_dtype)

        for i in range(ne):
            n = ns[i]
            if n == 0:
                Z[:, i] = X[:, i]
            if n < 0:
                s = (-1) ** np.abs(n)
                z0 = (X[:, i] - 1j * X[:, i + 1]) / np.sqrt(2)
                z1 = s * (X[:, i] + 1j * X[:, i + 1]) / np.sqrt(2)
                Z[:, i] = z0
                Z[:, i + 1] = z1

        return Z

    def rotate_multipliers(self, theta):

        b = np.zeros(self.ne, dtype=self.complex_dtype)
        for i in range(self.ne):
            b[i] = np.exp(1j * theta * self.ns[i])

        return b

    def lowpass(self, a, bandlimit):

        k = len(a) - 1
        for i in range(len(a)):
            if self.lmds[k] / (np.pi) > (bandlimit - 1) // 2:
                k = k - 1
        a[k + 1:] = 0
        return a

    #############################################################################
    #   Interpolation
    #############################################################################

    def barycentric_interp_sparse(self, x, xs, ys, s):
        # https://people.maths.ox.ac.uk/trefethen/barycentric.pdf

        n = len(x)
        m = len(xs)

        # Modify points by 2e-16 to avoid division by zero
        vals, x_ind, xs_ind = np.intersect1d(
            x, xs, return_indices=True, assume_unique=True
        )
        if self.dtype == np.float64:
            x[x_ind] = x[x_ind] + 2e-16
        else:
            x[x_ind] = x[x_ind] + 2e-8

        idx = np.zeros((n, s), dtype=self.dtype)
        jdx = np.zeros((n, s), dtype=self.dtype)
        vals = np.zeros((n, s), dtype=self.dtype)
        xss = np.zeros((n, s))
        idps = np.zeros((n, s), dtype=self.dtype)
        numer = np.zeros((n, 1), dtype=self.dtype)
        denom = np.zeros((n, 1), dtype=self.dtype)
        temp = np.zeros((n, 1), dtype=self.dtype)
        ws = np.zeros((n, s))
        xdiff = np.zeros(n)
        for i in range(n):

            # get a kind of blanced interval around our point
            k = np.searchsorted(x[i] < xs, True)

            idp = np.arange(k - s // 2, k + (s + 1) // 2)
            if idp[0] < 0:
                idp = np.arange(s)
            if idp[-1] >= m:
                idp = np.arange(m - s, m)
            xss[i, :] = xs[idp]
            jdx[i, :] = idp
            idx[i, :] = i

        x = x.reshape(-1, 1)
        Iw = np.ones(s, dtype=bool)
        ew = np.zeros((n, 1), dtype=self.dtype)
        xtw = np.zeros((n, s - 1), dtype=self.dtype)

        Iw[0] = False
        const = np.zeros((n, 1), dtype=self.dtype)
        for j in range(s):
            ew = np.sum(
                -np.log(np.abs(xss[:, 0].reshape(-1, 1) - xss[:, Iw])), axis=1
            )
            constw = np.exp(ew / s)
            constw = constw.reshape(-1, 1)
            const += constw
        const = const / s

        for j in range(s):
            Iw[j] = False
            xtw = const * (xss[:, j].reshape(-1, 1) - xss[:, Iw])
            ws[:, j] = 1 / np.prod(xtw, axis=1)
            Iw[j] = True

        xdiff = xdiff.flatten()
        x = x.flatten()
        temp = temp.flatten()
        denom = denom.flatten()
        for j in range(s):
            xdiff = x - xss[:, j]
            # if self.dtype == np.float32:
            #     xdiff = np.maximum(xdiff, 2e-8)

            temp = ws[:, j] / xdiff
            vals[:, j] = vals[:, j] + temp
            denom = denom + temp
        vals = vals / denom.reshape(-1, 1)

        vals = vals.flatten()
        idx = idx.flatten()
        jdx = jdx.flatten()
        A = spr.csr_matrix((vals, (idx, jdx)), shape=(n, m), dtype=self.dtype)
        A_T = spr.csr_matrix((vals, (jdx, idx)), shape=(m, n), dtype=self.dtype)

        return A, A_T

    def get_weights(self, xs):

        m = len(xs)
        I = np.ones(m, dtype=bool)
        I[0] = False
        e = np.sum(-np.log(np.abs(xs[0] - xs[I])))
        const = np.exp(e / m)
        ws = np.zeros(m, dtype=self.dtype)
        I = np.ones(m, dtype=bool)
        for j in range(m):
            I[j] = False
            xt = const * (xs[j] - xs[I])
            ws[j] = 1 / np.prod(xt)
            I[j] = True

        return ws


