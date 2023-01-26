
import numpy as np

class EigenPCA():

    def __init__(self, threshold = 0.9):
        self.eigen_vals = None
        self.eigen_vecs = None
        self.loadiings = None
        self.variance_explained = None
        self.cov_matrix = None
        self.threshold = threshold

    def fit(self, X):
        
        # input: X = data matrix with (n = n_samples, m = n_variables)
        n, m = X.shape
        self.mu = X.mean(axis=0, keepdims = True)
        
        # center the data
        X -= self.mu

        # compute covariance matrix
        C = np.cov(X.T) #np.dot(X.T, X) / (n-1)
        self.cov_matrix = C
        
        # eigen decomposition
        eigen_vals, eigen_vecs = np.linalg.eig(C)
        

        # sort by eigen values
        indices = np.argsort(eigen_vals)[::-1]
        self.eigen_vals = eigen_vals[indices]
        self.eigen_vecs = eigen_vecs[:,indices]

        # compute loadings and variance explained
        self.loadings = np.sqrt(self.eigen_vals) * self.eigen_vecs
        self.variance_explained = self.eigen_vals / np.sum(self.eigen_vals)  

        self.cumvar_explained = np.cumsum(self.variance_explained)

        # determine cut off based on variance explained
        if self.threshold >= 1.0:
            self.n_comp = m
        else:
            self.n_comp = np.argwhere(self.cumvar_explained > self.threshold)[0][0]

        return self.eigen_vals, self.eigen_vecs

    def project(self, X):
        
        # project data onto eigenvectors
        X -= self.mu
        P = self.eigen_vecs[:,:self.n_comp]

        X_proj = np.dot(X, P)
        return X_proj

    def reconstruct(self, X_proj):

        # reconstruct data from components
        P = self.eigen_vecs[:,:self.n_comp]
        X_rec = np.dot(X_proj, P.T) + self.mu
        return X_rec
        