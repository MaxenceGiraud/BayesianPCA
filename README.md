# Bayesian PCA

We implement the original PCA as a baseline, the probabilistic PCA [2] and Bayesian PCA [1]. We then also try to extend the bayesian formulation to the kernel PCA based also on the work on the Probabilistic Kernel PCA [3].

This project was done as part of a course on Bayesian Learning, taught by [Remi Bardenet](http://rbardenet.github.io/) as part of the [Master Data Science](https://sciences-technologies.univ-lille.fr/mathematiques/formation/master-mention-sciences-des-donnees/) at the University of Lille.

A notebook is available with the conducted experiments ([here](./experiments.ipynb)).


## Requirements 
* [NumPy](https://numpy.org/) 
* [SciPy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/) 

## Install

To install simply clone the project  :
```bash
git clone https://github.com/MaxenceGiraud/BayesianPCA
cd BayesianPCA
```

## Usage

```python
import bayespca as bpca

## Bayesian PCA
b = bpca.BayesianPCA()
b.fit_transform(X)

## Hinton diagram
bpca.utils.hinton(b.W.T)
```

## TODO
- [x] Implement original PCA using SVD and eigen decomposition
- [x] Probabilistic PCA (via eigendecomposition)
- [x] Implement EM algo to compute Probabilistic PCA
- [x] Bayesian PCA using EM
- [x] Kernel PCA
- [x] Probabilistic Kernel PCA (EM + via eigendecomposition)
- [x] Bayesian Kernel PCA

## References

[1] Bishop, C. M. (1999). Bayesian PCA. MIT Press.     
[2] Tipping, M. E . and C. M. Bishop (1997). Probabilistic principal component analysis. Journal of the Royal Statistical Society, B.          
[3] Z. Zhang, G. Wang, D.-Y. Yeung, and J. Kwok. Probabilistic kernel principal component analysis. 2004.      