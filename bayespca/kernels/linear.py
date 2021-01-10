import numpy as np
from .base_kernel import BaseKernel

class Linear(BaseKernel):    
    def __call__(self,x,y,**kwargs):
        x,y = self._reshape(x,y)
        
        return x @ y.T