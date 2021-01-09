# BayesianPCA

Implementation of Bayesian PCA based on the original paper by Bishop [1].

This project was done as part of a course on Bayesian Learning, taught by [Remi Bardenet](http://rbardenet.github.io/) as part of the [Master Data Science](https://sciences-technologies.univ-lille.fr/mathematiques/formation/master-mention-sciences-des-donnees/) at the University of Lille.

We implement the original PCA as a baseline, the probabilistic PCA [2] and Bayesian PCA [1]. We then also try to extend the bayesian formulation to the kernel PCA.

A notebook is available with the conducted experiments ([here](./experiments.ipynb)).

## Install

To install simply clone the project  :
```bash
git clone https://github.com/MaxenceGiraud/BayesianPCA
cd BayesianPCA
```

## Usage

```python
import bayespca as bpca

## Original PCA
p = bpca.PCA(n_components = 2)
p.fit_transform(X)
```

## TODO
- [x] Implement original PCA using SVD and eigen decomposition
- [x] Probabilistic PCA (via eigendecomposition)
- [x] Implement EM algo to compute Probabilistic PCA
- [x] Bayesian PCA using EM
- [x] Kernel PCA
- [ ] Probabilistic Kernel PCA
- [ ] Bayesian Kernel PCA

## References

[1] Bishop, C. M. (1999). Bayesian PCA. MIT Press.     
[2] Tipping, M. E . and C. M. Bishop (1997). Probabilistic principal component analysis. Journal of the Royal Statistical Society, B.         
