import numpy as np

class BaseKernel:

    def _reshape(self,x,y):
        ''' Reshape inputs x,y to 2D array if necessary'''
        if isinstance(x,int) or isinstance(x,float) or len(x.shape)<2 :
            x= np.array(x).reshape(-1,1)
        if isinstance(y,int) or isinstance(y,float) or len(y.shape)<2 :
            y= np.array(y).reshape(-1,1)
        return x,y