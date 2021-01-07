import numpy as np
from .kernels import RBF

class KernelPCA:
    ''' Kernel PCA
    Parameters
    ----------
    n_components : int,
        Number of components to consider
    kernel : callable,
        kernel used
    '''
    def __init__(self,n_components,kernel=RBF()):
        self.n_components = n_components
        self.kernel = kernel

    def _normalize(self,X,fit=True):
        if fit : 
            self.mean = np.mean(X,axis=0)
        return (X-self.mean)
    
    def _compute_ktild(self,X,X_train):
        K = self.kernel(X,X_train)
        O = (np.eye(X.shape[0]) @ np.eye(X_train.shape[0]).T )/ X_train.shape[0]
        K_tild = K - O @ K - K @ O + O @ K @O
        return  K_tild
    
    def _fit(self,X,transform=True):
       
        K_tild = self._compute_ktild(X,X) 

        eig_val,eig_vec = np.linalg.eig(K_tild) # Compute eigenvalues/vectors
        eig_sort = np.argsort(abs(eig_val))[::-1][:self.n_components] # Sort by eigenvalues and take the biggest n_components
        self.eigen_vec = np.real_if_close(eig_vec[:,eig_sort]/np.sqrt(eig_val[eig_sort]))

        if transform :
            return K_tild @ self.eigen_vec

    def fit(self,X):
        self.X = self._normalize(X,fit=True)
        self._fit(self.X,transform=False)

    def fit_transform(self,X):
        self.X = self._normalize(X,fit=True)
        return self._fit(self.X,transform=True)

    def transform(self,X):
        X = self._normalize(X,fit=False)
        K_tild = self._compute_ktild(X,self.X)
        return K_tild @ self.eigen_vec