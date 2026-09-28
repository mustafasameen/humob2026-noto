"""Adapted from xinychen/transdim imputer/TRMF.ipynb at eed7007ff49f83b12280844b209323e7383cc471.
MIT; Copyright (c) 2018 Xinyu Chen. See TRANSDIM_LICENSE and README.md.
"""
import numpy as np
from numpy.linalg import inv as inv


def TRMF(sparse_mat, init_para, init_hyper, time_lags, maxiter):
    """Temporal Regularized Matrix Factorization, TRMF."""

    ## Initialize parameters
    W = init_para["W"]
    X = init_para["X"]
    theta = init_para["theta"]

    ## Set hyperparameters
    lambda_w = init_hyper["lambda_w"]
    lambda_x = init_hyper["lambda_x"]
    lambda_theta = init_hyper["lambda_theta"]
    eta = init_hyper["eta"]

    dim1, dim2 = sparse_mat.shape
    observed = ~np.isnan(sparse_mat)
    sparse_mat = np.nan_to_num(sparse_mat, copy=True)
    d, rank = theta.shape

    for it in range(maxiter):
        ## Update spatial matrix W
        for i in range(dim1):
            pos0 = np.where(observed[i, :])
            Xt = X[pos0[0], :]
            vec0 = Xt.T @ sparse_mat[i, pos0[0]]
            mat0 = inv(Xt.T @ Xt + lambda_w * np.eye(rank))
            W[i, :] = mat0 @ vec0
        ## Update temporal matrix X
        for t in range(dim2):
            pos0 = np.where(observed[:, t])
            Wt = W[pos0[0], :]
            Mt = np.zeros((rank, rank))
            Nt = np.zeros(rank)
            if t < np.max(time_lags):
                Pt = np.zeros((rank, rank))
                Qt = np.zeros(rank)
            else:
                Pt = np.eye(rank)
                Qt = np.einsum('ij, ij -> j', theta, X[t - time_lags, :])
            if t < dim2 - np.min(time_lags):
                if t >= np.max(time_lags) and t < dim2 - np.max(time_lags):
                    index = list(range(0, d))
                else:
                    index = list(np.where((t + time_lags >= np.max(time_lags)) & (t + time_lags < dim2)))[0]
                for k in index:
                    Ak = theta[k, :]
                    Mt += np.diag(Ak ** 2)
                    theta0 = theta.copy()
                    theta0[k, :] = 0
                    Nt += np.multiply(Ak, X[t + time_lags[k], :]
                                      - np.einsum('ij, ij -> j', theta0, X[t + time_lags[k] - time_lags, :]))
            vec0 = Wt.T @ sparse_mat[pos0[0], t] + lambda_x * Nt + lambda_x * Qt
            mat0 = inv(Wt.T @ Wt + lambda_x * Mt + lambda_x * Pt + lambda_x * eta * np.eye(rank))
            X[t, :] = mat0 @ vec0
        ## Update AR coefficients theta
        for k in range(d):
            theta0 = theta.copy()
            theta0[k, :] = 0
            mat0 = np.zeros((dim2 - np.max(time_lags), rank))
            for L in range(d):
                mat0 += X[np.max(time_lags) - time_lags[L] : dim2 - time_lags[L] , :] @ np.diag(theta0[L, :])
            VarPi = X[np.max(time_lags) : dim2, :] - mat0
            var1 = np.zeros((rank, rank))
            var2 = np.zeros(rank)
            for t in range(np.max(time_lags), dim2):
                B = X[t - time_lags[k], :]
                var1 += np.diag(np.multiply(B, B))
                var2 += np.diag(B) @ VarPi[t - np.max(time_lags), :]
            theta[k, :] = inv(var1 + lambda_theta * np.eye(rank) / lambda_x) @ var2

        mat_hat = W @ X.T
        if (it + 1) % 100 == 0:
            print(f"TRMF iteration {it + 1}/{maxiter}", flush=True)

    return mat_hat
