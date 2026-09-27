import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class RobustPCA:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.L = None
        self.Y = np.zeros(self.D.shape)

        if mu is not None:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D.flatten(), ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda is not None:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum(np.abs(M) - tau, 0)

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, verbose=False, iter_print=100):
        """
        Parameters
        ----------
        tol : float or None
            Convergence tolerance. Defaults to 1e-7 * ||D||_F.
        max_iter : int
            Maximum number of ADMM iterations.
        verbose : bool, default False
            If True, prints iteration/error progress. If False (default),
            fit() runs silently regardless of iter_print.
        iter_print : int
            Print every `iter_print` iterations, only when verbose=True.
        """
        n_iter = 0
        err = np.inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol is not None:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        # Principal Component Pursuit by Alternating Directions (ADMM)
        # Algorithm 1 from Candès et al., 2011 (https://dl.acm.org/doi/10.1145/1970392.1970395)
        while err > _tol and n_iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)                # step 3
            Sk = self.shrink(
                self.D - Lk + self.mu_inv * Yk, self.lmbda * self.mu_inv)   # step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)                         # step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            n_iter += 1
            if verbose and (
                (n_iter % iter_print) == 0 or n_iter == 1
                or n_iter > max_iter or err <= _tol
            ):
                print(f'iteration: {n_iter}, error: {err}')

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def pca(self, n_components=None, standardize=True):
        """
        Compute PCA on the low-rank component L obtained from fit(),
        using scikit-learn's PCA implementation.

        Parameters
        ----------
        n_components : int, float, or None
            Passed directly to sklearn.decomposition.PCA. If None,
            sklearn keeps min(n_samples, n_features) components.
        standardize : bool, default True
            If True, standardizes L's columns (zero mean, unit variance)
            before PCA via StandardScaler.
            If False, only centers the data (classical PCA on covariance).
        """
        if self.L is None:
            raise RuntimeError("You must call fit() before pca().")

        if standardize:
            X = StandardScaler().fit_transform(self.L)
        else:
            X = self.L - self.L.mean(axis=0)

        model = PCA(n_components=n_components)
        scores = model.fit_transform(X)

        return {
            'scores': scores,
            'loadings': model.components_.T,
            'explained_variance': model.explained_variance_,
            'explained_variance_ratio': model.explained_variance_ratio_,
            'n_components': model.n_components_,
            'pca_model': model,
        }