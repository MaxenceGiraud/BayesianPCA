import numpy as np

class ProbabilisticKernelPCA:
    def __init__(self,n_components, kernel,method = 'eig',max_iter=100):
        if method not in ["eig","em"]:
            raise ValueError("method must be either eig or em")

        self.q = n_components
        self.kernel = kernel
        self.max_iter = max_iter
        self.method = method
    
    def _fit_em(self):

        # Init
        N,self.d = self.X.shape
        self.sigma2 = np.random.random()
        self.W = np.random.randn(N, self.q)
        
        H = np.eye(N) - 1/N * np.ones(N).reshape(-1,1) @ np.ones(N).reshape(1,-1)
        self.K = self.kernel(self.X,self.X)
        self.M_inv = np.linalg.inv(self.W.T @ self.W + self.sigma2 + np.eye(self.q))


        # EM  algorithm
        i = 0
        while i < self.max_iter :
            old_W = self.W
          
            self.W = H @ self.K @ H @ self.W @ np.linalg.inv( self.M_inv @ self.W.T @ H @ self.K @ H @ self.W)
            # start inv :  self.d * self.sigma2 * np.eye(self.q) +
            self.sigma2 = 1/N**2 * np.trace(H @ self.K @ H - H @ self.K @ H @ old_W @ self.M_inv @ self.W.T)

            self.M_inv = np.linalg.inv(self.W.T @ self.W + self.sigma2 + np.eye(self.q))

            i+=1

    def _fit_eig_decomp(self):
        N,self.d = self.X.shape

        H = np.eye(N) - 1/N * np.ones(N).reshape(-1,1) @ np.ones(N).reshape(1,-1)
        self.K = self.kernel(self.X,self.X)

        S = 1/N * H @ self.K @ H # Sample Covariance Matrix
        eig_val,eig_vec = np.linalg.eig(S) # Compute eigenvalues/vectors
        eig_sort = np.argsort(eig_val)[::-1]# Sort by eigenvalues and take the biggest n_components

        self.sigma2 = 1/(self.d-self.q) * np.sum(eig_val[eig_sort[-self.q:]])

        self.W = eig_vec[:,eig_sort[:self.q]] @ np.sqrt(np.diag(eig_val[eig_sort[:self.q]])-self.sigma2*np.eye(self.q))
        self.W = np.real_if_close(self.W) # Take real part if casted to complex values (only 0 imaginary part, simply the type of the object)

        self.M_inv = np.real_if_close(np.linalg.inv(self.W.T @ self.W + self.sigma2 + np.eye(self.q)))



    def fit(self,X):    

        # Capture the mean
        self.mu = X.mean(axis=0)
        self.X = X - self.mu

        if self.method == 'eig':
            self._fit_eig_decomp()
        elif self.method == 'em':
            self._fit_em()

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X,K=self.K)


    def transform(self,X,K=None):
        X_transform = np.zeros((X.shape[0],self.q))
        
        if K is None :
            K = self.kernel(self.X,(X-self.mu))

        # Parameters of the Gaussians
        transform_cov = self.sigma2 *  self.M_inv
        transform_means = self.M_inv @ self.W.T @ K

        for i in range(X.shape[0]):
            X_transform[i] = np.random.multivariate_normal(transform_means[:,i],transform_cov)

        return X_transform