import numpy as np


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

    def fit(self, tol=None, max_iter=1000, iter_print=100):
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
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)                            #this line implements step 3
            Sk = self.shrink(
                self.D - Lk + self.mu_inv * Yk, self.lmbda * self.mu_inv)               #this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)                                      #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            n_iter += 1
            if (n_iter % iter_print) == 0 or n_iter == 1 or n_iter > max_iter or err <= _tol:
                print(f'iteration: {n_iter}, error: {err}')

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def pca(self, n_components=None, tol=1e-6):
        """
        Compute PCA (via SVD) on the low-rank component L obtained from fit().

        Parameters
        ----------
        n_components : int or None
            Number of components to retain.
            If None, the effective rank of L is used.
        tol : float
            Threshold to determine the effective rank of L.

        Returns
        -------
        dict with keys:
            'scores' : ndarray, shape (n_samples, k)
            'loadings' : ndarray, shape (n_features, k)
            'explained_variance' : ndarray, shape (k,)
            'explained_variance_ratio' : ndarray, shape (k,)
            'n_components' : int
        """
        if self.L is None:
            raise RuntimeError("You must call fit() before pca().")

        # Center L (classical PCA centers the data)
        L_centered = self.L - self.L.mean(axis=0)

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(L_centered, full_matrices=False)

        # Determine the number of components
        if n_components is None:
            k = int(np.sum(S > tol * S[0]))
        else:
            k = n_components

        # Truncate to the first k components
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]

        # Compute scores and loadings
        scores = U_k * S_k
        loadings = Vt_k.T

        # Compute explained variance
        n = self.L.shape[0]
        explained_variance = (S_k ** 2) / (n - 1)
        total_var = np.sum(S ** 2) / (n - 1)
        explained_variance_ratio = explained_variance / total_var

        return {
            'scores': scores,
            'loadings': loadings,
            'explained_variance': explained_variance,
            'explained_variance_ratio': explained_variance_ratio,
            'n_components': k,
        }