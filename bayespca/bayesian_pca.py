import numpy as np

class BayesianPCA:
    def __init__(self,max_iter=100,eps=1e-6):
        self.max_iter = max_iter
        self.eps = eps
    
    def _init_params(self,X):
        sample_cov = 1/X.shape[0] * X.T @ X
        eig_val,eig_vec = np.linalg.eig(sample_cov) # Compute eigenvalues/vectors
        eig_sort = np.argsort(eig_val)[::-1]# Sort by eigenvalues and take the biggest n_components

        self.sigma2 = eig_val.min()

        self.W = eig_vec[:,eig_sort[:self.q]] @ np.sqrt(np.diag(eig_val[eig_sort[:self.q]])-self.sigma2*np.eye(self.q))


    def _expectation_step(self,X):
        # Evaluation of the expected sufficient statistics of the latent-space posterior distribution
        self.M = self.W.T @ self.W +self.sigma2 * np.eye(self.q)
        self.x = np.linalg.pinv(self.M) @ self.W.T @ X.T
        self.xxt = self.sigma2 * self.M + self.x    @ self.x.T

    def _maximization_step(self,X):
        # Update of the model parameters
        self.W =(X.T @ self.x.T) @ np.linalg.pinv(np.sum(self.xxt ,axis=0) + self.sigma2 * np.diag(self.alpha)) 
        self.sigma2 = (np.linalg.norm(X) - 2 * np.sum(self.x.T @ self.W.T @ X.T) + np.trace(self.xxt @ self.W.T @ self.W)) / (X.shape[0]*X.shape[1]) 
    
        self.alpha = self.d / np.linalg.norm(self.W,axis=0) # Re-estimation of alphas

    def fit(self,X):
        self.d = X.shape[1]
        self.q = self.d - 1 

        self.mu = np.mean(X,axis=0) 
        X = X - self.mu

        self._init_params(X)
        self.alpha = self.d / np.linalg.norm(self.W,axis=0) # Re-estimation of alphas

        i = 0
        old_norm_w = np.zeros(self.q)
        while i < self.max_iter and np.any(abs(np.linalg.norm(self.W,axis=0) - old_norm_w) > self.eps):
            print(i)
            
            old_norm_w = np.linalg.norm(self.W,axis=0)

            # Expectation step
            self._expectation_step(X)

            # Maximization step
            self._maximization_step(X)

            i+=1

        # Find effective dimensionality
        sum_alpha = np.sum(1/self.alpha)
        self.eff_dim = np.array([i for i, inv_alpha in enumerate(1/self.alpha) if inv_alpha < sum_alpha/self.q])
        self.qeff = len(self.eff_dim)

        self.M_inv = np.linalg.pinv(self.W[:,self.eff_dim].T @ self.W[:,self.eff_dim] + self.sigma2 * np.eye(self.qeff))
        # zero_norm = np.where(np.linalg.norm(self.W.T,axis=0) >1e-10,1,0)
        # self.qeff = zero_norm.sum()

        # self.W = self.W[:,1-zero_norm]
        # self.alpha = self.alpha[1-zero_norm]

    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    def transform(self,X):
        
        Z = self.M_inv @ self.W[:,self.eff_dim].T @ (X-self.mu).T
        return Z