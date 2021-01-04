# BayesianPCA

Implementation of Bayesian PCA based on the original paper by Bishop [1].

This project was done as part of a course on Bayesian Learning, taught by [Remi Bardenet](http://rbardenet.github.io/) as part of the [Master Data Science](https://sciences-technologies.univ-lille.fr/mathematiques/formation/master-mention-sciences-des-donnees/) at the University of Lille.

We implement the original PCA as a baseline, the probabilistic PCA [2] and Bayesian PCA [1].

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
- [ ] Implement EM algo to compute Probabilistic PCA
- [ ] Bayesian PCA
- [ ] Build a mixture of probabilistic PCA
- [ ] Build a mixture of Bayesian PCA

## References

[1] Bishop, C. M. (1999). Bayesian PCA. MIT Press.     
[2] Tipping, M. E . and C. M. Bishop (1997). Probabilistic principal component analysis. Journal of the Royal Statistical Society, B.          
[3] Tipping, M. E. and C. M. Bishop (1997). Mixtures of principal component analysers. In Proceedings lEE Fifth International Conference on Artificial Neural Networks.Cambridge, U.K., July. , pp. 13-18.