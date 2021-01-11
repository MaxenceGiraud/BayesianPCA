import numpy as np
from .kernels import RBF

class BayesianKernelPCA:
    def __init__(self, kernel=RBF(),max_iter=100):
    
        self.kernel = kernel
        self.max_iter = max_iter
    
    def _fit_em(self):

        # Init
        N,self.d = self.X.shape
        self.q = self.d - 1

        self.sigma2 = np.random.random()
        self.W = np.random.randn(N, self.q)
        self.alpha = np.zeros(self.q)
        
        H = np.eye(N) - 1/N * np.ones(N).reshape(-1,1) @ np.ones(N).reshape(1,-1)
        self.K = self.kernel(self.X,self.X)
        self.M_inv = np.linalg.inv(self.W.T @ self.W + self.sigma2 + np.eye(self.q))


        # EM  algorithm
        i = 0
        while i < self.max_iter :
            old_W = self.W
          
            self.W = H @ self.K @ H @ self.W @ np.linalg.inv(self.M_inv @ self.W.T @ H @ self.K @ H @ self.W + N * self.sigma2 * (np.diag(self.alpha)+np.eye(self.q)))

            self.sigma2 = 1/N**2 * np.trace(H @ self.K @ H - H @ self.K @ H @ old_W @ self.M_inv @ self.W.T)

            self.M_inv = np.linalg.inv(self.W.T @ self.W + self.sigma2 + np.eye(self.q))

            self.alpha = self.d / np.linalg.norm(self.W,axis=0)**2 # Re-estimation of alphas

            i+=1


    def fit(self,X):    

        # Capture the mean
        self.mu = X.mean(axis=0)
        self.X = X - self.mu

        self._fit_em()

        sum_alpha = np.sum(1/self.alpha)
        self.eff_dim = np.array([i for i, inv_alpha in enumerate(1/self.alpha) if inv_alpha < sum_alpha/self.q])
        self.qeff = len(self.eff_dim)


    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X,K=self.K)

    def transform(self,X,K=None):
        X_transform = np.zeros((X.shape[0],self.qeff))
        
        if K is None :
            K = self.kernel(self.X,(X-self.mu))

        # Parameters of the Gaussians
        transform_cov = self.sigma2 *  self.M_inv[self.eff_dim][:,self.eff_dim]
        transform_means = (self.M_inv[self.eff_dim][:,self.eff_dim] @ self.W.T[self.eff_dim] @ K).reshape(-1,X.shape[0])

        for i in range(X.shape[0]):
            X_transform[i] = np.random.multivariate_normal(transform_means[:,i],transform_cov)

        return X_transform