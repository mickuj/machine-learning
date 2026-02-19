#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer()


# In[2]:


from sklearn.datasets import load_iris
data_iris = load_iris()


# In[3]:


X_bc = data_breast_cancer.data
X_ir = data_iris.data

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

pca_bc = PCA(n_components=0.9)
X_bc_reduced = pca_bc.fit_transform(X_bc)
print("bc - liczba wymiarów:", X_bc_reduced.shape[1])
print("bc - zmienność:", pca_bc.explained_variance_ratio_)

pca_ir = PCA(n_components=0.9)
X_ir_reduced = pca_ir.fit_transform(X_ir)
print("ir - liczba wymiarów:", X_ir_reduced.shape[1])
print("ir - zmienność:", pca_ir.explained_variance_ratio_)


# In[4]:


X_bc_scaled = StandardScaler().fit_transform(X_bc)
X_ir_scaled = StandardScaler().fit_transform(X_ir)

pca_bc_scaled = PCA(n_components=0.9)
X_bc_scaled_reduced = pca_bc_scaled.fit_transform(X_bc_scaled)
print("bc_scaled - liczba wymiarów:", X_bc_scaled_reduced.shape[1])
print("bc_scaled - zmienność:", pca_bc_scaled.explained_variance_ratio_)

pca_ir_scaled = PCA(n_components=0.9)
X_ir_scaled_reduced = pca_ir_scaled.fit_transform(X_ir_scaled)
print("ir_scaled - iczba wymiarów:", X_ir_scaled_reduced.shape[1])
print("ir_scaled - zmienność:", pca_ir_scaled.explained_variance_ratio_)


# In[5]:


with open('pca_bc.pkl', 'wb') as f:
    pickle.dump(pca_bc_scaled.explained_variance_ratio_, f)

with open('pca_ir.pkl', 'wb') as f:
    pickle.dump(pca_ir_scaled.explained_variance_ratio_, f)


# In[6]:


import numpy as np

def oblicz_cechy(pca):
    weights = np.abs(pca.components_ * pca.explained_variance_ratio_[:, np.newaxis])
    idx = np.argsort(np.max(weights, axis=0))[::-1]
    return idx

idx_bc = oblicz_cechy(pca_bc_scaled)
with open('idx_bc.pkl', 'wb') as f:
    pickle.dump(idx_bc, f)

idx_ir = oblicz_cechy(pca_ir_scaled)
with open('idx_ir.pkl', 'wb') as f:
    pickle.dump(idx_ir, f)

print("breast_can:", idx_bc)
print("iris:", idx_ir)

