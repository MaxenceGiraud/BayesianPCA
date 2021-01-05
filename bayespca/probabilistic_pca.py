import numpy as np
from sklearn.exceptions import NotFittedError

class ProbabilisticPCA:
    ''' Probabilistic PCA 

    Parameters
    ----------
    n_components : int,
        Number of components to consider
    method : str,
        Method of computation (either eig for Eigen Decomposition or em for Expectation-Maximization algorithm), Default is eigen decomposition
    '''
    def __init__(self,n_components,method = "eig"):
        if method not in ["eig","em"]:
            raise ValueError("method must be either eig or em")
        if method == "em":
            raise NotImplementedError

        self.n_components = n_components
        self.method = method

    def fit(self,X):
        assert X.shape[1] > self.n_components, "Number of components must be at least d-1"
        # Normalize data
        self.mean =  X.mean(axis=0)
        X = (X -self.mean) 

        if self.method == 'eig':
            self._fit_eig_decomp(X)
        elif self.method == 'em':
            self._fit_em(X)

    def _expectation_step(self):
        raise NotImplementedError

    def _maximization_step(self):
        raise NotImplementedError

    def _fit_em(self,X):
        iter_max = 100
        for _ in range(iter_max):
            self._expectation_step()
            self._maximization_step()

    def _fit_eig_decomp(self,X):
        S = 1/X.shape[0] * X.T @ X # Sample Covariance Matrix
        eig_val,eig_vec = np.linalg.eig(S) # Compute eigenvalues/vectors
        eig_sort = np.argsort(eig_val)[::-1]# Sort by eigenvalues and take the biggest n_components

        self._sigma2 = 1/(X.shape[-1]-self.n_components) * np.sum(eig_val[eig_sort[-self.n_components:]])

        self.W = eig_vec[:,eig_sort[:self.n_components]] @ np.sqrt(np.diag(eig_val[eig_sort[:self.n_components]])-self._sigma2*np.eye(self.n_components))

        self.C = self.W @ self.W.T + self._sigma2 * np.eye(X.shape[1]) # Observation Covariance
        self.M_inv = np.linalg.pinv(self.W.T @ self.W + self._sigma2 * np.eye(self.n_components))
    

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    def transform(self,X):
        X_transform = np.zeros((X.shape[0],self.n_components))

        # Parameters of the Gaussians
        transform_cov = self._sigma2 *  self.M_inv
        transform_means = self.M_inv @ self.W.T @ (X-self.mean).T

        for i in range(X.shape[0]):
            X_transform[i] = np.random.multivariate_normal(transform_means[:,i],transform_cov)

        return X_transform

    def generate(self,n_samples):
        return np.random.multivariate_normal(self.mean,self.C,size=n_samples)
    
    def generate_transform(self,n_samples):
        return self.transform(self.generate(n_samples=n_samples))
    
    def _compute_log_likelihood(self,X):
        S = 1/X.shape[0] * (X-self.mean).T @ (X-self.mean) # Sample Covariance matrix
        l = np.sum(-X.shape[0]/2 * (X.shape[1]*np.log(2*np.pi) + np.log(abs(self.C) +np.trace(np.linalg.inv(self.C)@S)) ))

        return l 