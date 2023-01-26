
from pca import EigenPCA

import numpy as np
import sklearn.decomposition

X = np.random.random((100,30))

pca_model = EigenPCA(threshold=1.0)

eig_vals, eig_vec = pca_model.fit(X)
X_proj1 = pca_model.project(X)
Xhat1 = pca_model.reconstruct(X_proj1)

print(X[1,:])
print(Xhat1[1,:])
# print(pca_model.eigen_vecs[:,1])
# print(X_proj1[1,:])

pca = sklearn.decomposition.PCA()
pca.fit(X)

mu = np.mean(X, axis=0)
nComp = pca_model.n_comp
X_proj2 = pca.transform(X)[:,:nComp]
Xhat2 = np.dot(X_proj2, pca.components_[:nComp,:])
Xhat2 += mu

# print(pca.components_[1,:])
# 
# print(X_proj2[1,:])
# print(X[1,:])