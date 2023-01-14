from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import numpy as np
import sys
import pickle

def logit_regress(X,y,train_idx,regul=1):

    X_train = X[:,train_idx].T
    y_train = y[train_idx]

    model = LogisticRegressionCV(cv=10,n_jobs=12,max_iter=5000,tol=1e-3)
    model.fit(X_train, y_train)

    return model

def get_roc(model,testx,testy):
    # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    
    ns_probs = [0 for _ in range(len(testy))]
    lr_probs = model.predict_proba(testx)
    lr_probs = lr_probs[:, 1]

    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)

    #summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)

    return ns_fpr, ns_tpr, lr_fpr, lr_tpr

def kl_divergence(p, q):
    if p.sum()!=0:
        p = p / p.sum(keepdims=True)
    
    if q.sum()!=0:
        q = q / q.sum(keepdims=True)
        
    return np.sum(np.where(p > 1e-9, p * np.log(p / q), 0))



def pca(X,n_comp=3):
  # Data matrix X, assumes 0-centered
  n, m = X.shape

#   if not np.allclose(X.mean(axis=0), np.zeros(m)):
#     X -= X.mean(axis=0,keepdims=True)
  # Compute covariance matrix
  C = np.dot(X.T, X) / (n-1)
  # Eigen decomposition
  eigen_vals, eigen_vecs = np.linalg.eig(C)
  
  return eigen_vals.T, eigen_vecs, (eigen_vecs.T[:][:n_comp])

def show_confusion_matrix(x,y,model,ax):
    cm = confusion_matrix(y, model.predict(x))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Convex\n(predicted)', 'Concave\n(predicted)'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Convex\n(actual)', 'Concave\n(actual)'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='k',fontsize=10)