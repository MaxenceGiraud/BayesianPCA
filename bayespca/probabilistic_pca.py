import numpy as np

class ProbabilisticPCA:
    ''' Probabilistic PCA 

    Parameters
    ----------
    n_components : int,
        Number of components to consider
    method : str,
        Method of computation (either eig for Eigen Decomposition or em for Expectation-Maximization algorithm), Default is eigen decomposition
    max_iter : int,
        Maximum number of iteration for EM algo (ignored if method == "eig")
    '''
    def __init__(self,n_components,method = "eig",max_iter=200):
        if method not in ["eig","em"]:
            raise ValueError("method must be either eig or em")

        self.n_components = n_components
        self.method = method
        self.max_iter = max_iter
    
    def _init_params_em(self,X):
        self.sigma2 = np.random.random()
        self.W = np.random.randn(X.shape[1], self.n_components)

    def fit(self,X):
        assert X.shape[1] > self.n_components, "Number of components must be at least d-1"
        # Normalize data
        self.mean =  X.mean(axis=0)
        X = (X -self.mean) 

        if self.method == 'eig':
            self._fit_eig_decomp(X)
        elif self.method == 'em':
            self._fit_em(X)
        
        self._compute_final_params(X.shape[1])

    def _expectation_step(self,X):
        # Evaluation of the expected sufficient statistics of the latent-space posterior distribution
        self.M = self.W.T @ self.W +self.sigma2 * np.eye(self.n_components)
        self.M_inv = np.linalg.pinv(self.M)
        self.x = self.M_inv @ self.W.T @ X.T
        self.xxt = self.sigma2 * self.M_inv + self.x @ self.x.T

    def _maximization_step(self,X):
        old_w = self.W
        # Update of the model parameters
        self.W =(X.T @ self.x.T) @ np.linalg.pinv(self.xxt)

        self.sigma2 = np.trace(X +  X @ old_w @ self.M_inv @ self.W.T)/ (X.shape[0]*X.shape[1]) 
    

    def _fit_em(self,X):
        self._init_params_em(X)
        w_norm_diff = 1
        i= 0
        while i< self.max_iter and w_norm_diff > 1e-6 : 
            W_old = self.W
            self._expectation_step(X)
            self._maximization_step(X)

            w_norm_diff = np.linalg.norm(self.W-W_old)
            i+=1

    def _fit_eig_decomp(self,X):
        S = 1/X.shape[0] * X.T @ X # Sample Covariance Matrix
        eig_val,eig_vec = np.linalg.eig(S) # Compute eigenvalues/vectors
        eig_sort = np.argsort(eig_val)[::-1]# Sort by eigenvalues and take the biggest n_components

        self.sigma2 = 1/(X.shape[-1]-self.n_components) * np.sum(eig_val[eig_sort[-self.n_components:]])

        self.W = eig_vec[:,eig_sort[:self.n_components]] @ np.sqrt(np.diag(eig_val[eig_sort[:self.n_components]])-self.sigma2*np.eye(self.n_components))

    
    def _compute_final_params(self,d):
        self.C = self.W @ self.W.T + self.sigma2 * np.eye(d) # Observation Covariance
        self.M_inv = np.linalg.pinv(self.W.T @ self.W + self.sigma2 * np.eye(self.n_components))
    

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    def transform(self,X):
        X_transform = np.zeros((X.shape[0],self.n_components))

        # Parameters of the Gaussians
        transform_cov = self.sigma2 *  self.M_inv
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
        l = np.sum(-X.shape[0]/2 * (X.shape[1]*np.log(2*np.pi) + np.log(np.linalg.det(self.C) +np.trace(np.linalg.inv(self.C)@S)) ))

        return l 