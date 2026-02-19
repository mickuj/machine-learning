#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# In[2]:


from sklearn.cluster import KMeans, DBSCAN
import pickle
from sklearn.metrics import silhouette_score

silh_scores = []

for k in [8, 9, 10, 11, 12]:
    best_inertia = float('inf')
    best_k = None
    for _ in range(10):
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(X)
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_k = kmeans
    silh_scores.append(silhouette_score(X, best_k.labels_))

silh_scores


# In[3]:


with open('kmeans_sil.pkl', 'wb') as f:
    pickle.dump(silh_scores, f)


# In[4]:


kmeans_10 = KMeans(n_clusters=10, n_init=10)
kmeans_10.fit(X)
silh_score_10 = silhouette_score(X, kmeans_10.labels_)
silh_score_10


# In[5]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, kmeans_10.labels_)
cm


# In[6]:


max_i = [np.argmax(row) for row in cm]
i_sorted = sorted(set(max_i))
i_sorted


# In[7]:


with open('kmeans_argmax.pkl', 'wb') as f:
    pickle.dump(i_sorted, f)


# In[8]:


distances = []
X_300 = X[:300]

for i in range(300):
    for j in range(i+1, len(X)):
        dist = np.linalg.norm(X_300[i] - X[j])
        distances.append(dist)
sorted_dist = sorted(distances)
sorted_dist[:10]


# In[9]:


with open('dist.pkl', 'wb') as f:
    pickle.dump(sorted_dist[:10], f)


# In[11]:


s = np.mean(sorted_dist[:3])
eps_val = np.arange(s, s+0.1*s, step=0.04*s)
eps_val


# In[13]:


from sklearn.cluster import DBSCAN

db_labels = []
for eps in eps_val:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan.fit(X)
    db_labels.append(len(set(dbscan.labels_)))

db_labels


# In[14]:


with open('dbscan_len.pkl', 'wb') as f:
    pickle.dump(db_labels, f)

