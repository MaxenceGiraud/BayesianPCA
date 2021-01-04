import numpy as np

class PCA:
    ''' Original Principal Component analysis, can be computed using the 2 standards method : SVD or eigenvectors of the covariance

    Parameters
    ----------
    n_components : int,
        Number of components to consider
    method : str,
        Method of computation (either svd of cov), both methods yield the same results
    '''

    def __init__(self,n_components,method='svd'):
        self.n_components = n_components
        assert method in ['svd','cov'], "PCA method must be either svd or cov"
        self.method = method

    def _fit(self,X):
        if self.method == 'svd' :
            return np.linalg.svd(X)

        elif self.method == 'cov' : 
            cov = (X.T@X) /(X.shape[0]-1) # Covariance matrix
            eig_val,eig_vec = np.linalg.eig(cov) # Compute eigenvalues/vectors
            eig_sort = np.argsort(eig_val)[::-1][:self.n_components] # Sort by eigenvalues and take the biggest n_components

            return eig_vec[:,eig_sort]

    def fit(self,X):
        if self.method == 'svd' :
            _,s,_ = self._fit(X)
            self.singular_values = s[:self.n_components]
        
        elif self.method == 'cov' : 
            self.eigen_vec = self._fit(X)

    def fit_transform(self,X):
        # Normalize data
        self.mean =  X.mean(axis=0)
        X = (X -self.mean)

        if self.method == 'svd' :
            u,s,_ = self._fit(X)
            self.singular_values  = s[:self.n_components]
            return u[:,:self.n_components] * self.singular_values

        elif self.method == 'cov' :
            
            self.eigen_vec = self._fit(X)
            return X @ self.eigen_vec


    def transform(self,X):
        X = (X-self.mean) 
        if self.method == 'svd':
            u,_,_ = self._fit(X)
            return u[:,self.n_components] * self.singular_values
        
        elif self.method == 'cov' :
            return X @ self.eigen_vec