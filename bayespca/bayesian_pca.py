import numpy as np

class BayesianPCA:
    def __init__(self,max_iter=100,eps=1e-6):
        self.max_iter = max_iter
        self.eps = eps

    def _init_params(self,X):
        self.sigma2 = np.random.random()
        self.W = np.random.randn(self.d, self.q)
        self.alpha = self.d / np.linalg.norm(self.W,axis=0) # Estimation of alphas

    def _expectation_step(self,X):
        # Evaluation of the expected sufficient statistics of the latent-space posterior distribution
        self.M = self.W.T @ self.W +self.sigma2 * np.eye(self.q)
        self.M_inv = np.linalg.pinv(self.M)
        self.x = self.M_inv @ self.W.T @ X.T
        self.xxt = self.sigma2 * self.M_inv + self.x @ self.x.T # (inv M ??)

    def _maximization_step(self,X):
        old_w = self.W
        # Update of the model parameters
        self.W =(X.T @ self.x.T) @ np.linalg.pinv(self.xxt +self.sigma2 * (np.diag(self.alpha) + np.eye(self.q))) 
        
        self.sigma2 = np.trace(X +  X @ old_w @ self.M_inv @ self.W.T)/ (X.shape[0]*X.shape[1]) 
    
        self.alpha = self.d / np.linalg.norm(self.W,axis=0)**2 # Re-estimation of alphas

    def fit(self,X):
        self.d = X.shape[1]
        self.q = self.d - 1

        self.mu = np.mean(X,axis=0) 
        X = X - self.mu

        self._init_params(X)

        i = 0
        old_norm_w = np.zeros(self.q)
        while i < self.max_iter :# and np.any(abs(np.linalg.norm(self.W,axis=0) - old_norm_w) > self.eps):
            
            old_norm_w = np.linalg.norm(self.W,axis=0)

            # Expectation step
            self._expectation_step(X)

            # Maximization step
            self._maximization_step(X)

            i+=1

        sum_alpha = np.sum(1/self.alpha)
        self.eff_dim = np.array([i for i, inv_alpha in enumerate(1/self.alpha) if inv_alpha < sum_alpha/self.q])
        self.qeff = len(self.eff_dim)

        # Compute params for transform
        self.M_inv = np.linalg.pinv(self.W[:,self.eff_dim].T @ self.W[:,self.eff_dim] + self.sigma2 * np.eye(self.qeff))

    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    def transform(self,X):
        self.M_inv = np.linalg.pinv(self.W[:,self.eff_dim].T @ self.W[:,self.eff_dim] + self.sigma2 * np.eye(self.qeff))
        # Parameters of the Gaussians
        transform_cov = self.sigma2 *  self.M_inv
        transform_means = self.M_inv @ self.W[:,self.eff_dim].T @ (X-self.mu).T

        X_transform = np.zeros((X.shape[0],self.qeff))
        for i in range(X.shape[0]):
            X_transform[i] = np.random.multivariate_normal(transform_means[:,i],transform_cov)

        return X_transform
    
    def _compute_log_likelihood(self,X):
        # TODO remake this (this is only the first part of the log likelihood)
        C = self.W @ self.W.T + self.sigma2 * np.eye(X.shape[1]) # Observation Covariance
        S = 1/X.shape[0] * (X-self.mu).T @ (X-self.mu) # Sample Covariance matrix
        l = -X.shape[0]/2 * (X.shape[1]*np.log(2*np.pi) + np.log(np.linalg.det(C) +np.trace(np.linalg.inv(C)@S)) )

        return l 