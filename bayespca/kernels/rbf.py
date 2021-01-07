import numpy as np
from scipy.spatial.distance import cdist
from .base_kernel import BaseKernel

class RBF(BaseKernel):
    def __init__(self,l=1):
        self.l =l

    def __call__(self,x,y):
        x,y = self._reshape(x,y)
        dist = cdist(x,y)
        return  np.exp(-0.5*dist**2/self.l**2 ) 