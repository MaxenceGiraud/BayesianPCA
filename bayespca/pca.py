import numpy as np

class PCA:
    def __init__(self,n_components):
        self.n_components = n_components

    def _fit(self,X):
        # Center data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        return np.linalg.svd(X)

    def fit(self,X):
        _,s,_ = self._fit(X)
        self.singular_values = s[:self.n_components]

    def fit_transform(self,X):
        u,s,_ = self._fit(X)
        self.singular_values  = s[:self.n_components]
        return u[:,:self.n_components] * self.singular_values


    def transform(self,X):
        u,_,_ = self._fit(X)
        return u[:,self.n_components] * self.singular_values