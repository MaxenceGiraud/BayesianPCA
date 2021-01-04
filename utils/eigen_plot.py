import matplotlib.pyplot as plt
import numpy as np 

def plot_eigenvec(X,eigenvectors,labels=None,feature_names=None):
    if labels is None :
        labels = np.zeros(X.shape[0])
    if feature_names is None :
        feature_names =np.arange(eigenvectors.shape[0])
    
    plt.scatter(X[:,0],X[:,1],c=labels)
    for i in range(eigenvectors.shape[0]):
        plt.arrow(0,0,eigenvectors[i,0],eigenvectors[i,1],color='black', head_width=0.05, head_length=0.1)
        plt.text(eigenvectors[i,0]*1.3,eigenvectors[i,1]*1.3,feature_names[i],color='black')
        
    plt.show()