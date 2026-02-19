#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=False)
print(data_breast_cancer['DESCR'])


# In[2]:


data_iris = datasets.load_iris()
print(data_iris['DESCR'])


# In[3]:


from sklearn.model_selection import train_test_split
X = data_breast_cancer.data[:, [3, 4]]
y = data_breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
clf1 = LinearSVC(loss="hinge", random_state=42, max_iter=10000)
clf1.fit(X_train, y_train)
y_pred_train1 = clf1.predict(X_train)
y_pred_test1 = clf1.predict(X_test)
acc_train1 = accuracy_score(y_train, y_pred_train1)
acc_test1 = accuracy_score(y_test, y_pred_test1)


# In[5]:


from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(StandardScaler(), LinearSVC(loss="hinge", random_state=42, max_iter=10000))
pipeline.fit(X_train, y_train)
y_pred_train2 = pipeline.predict(X_train)
y_pred_test2 = pipeline.predict(X_test)
acc_train2 = accuracy_score(y_train, y_pred_train2)
acc_test2 = accuracy_score(y_test, y_pred_test2)


# In[6]:


import pickle 
accs = [acc_train1, acc_test1, acc_train2, acc_test2]
with open('bc_acc.pkl', 'wb') as f:
    pickle.dump(accs, f)
accs


# In[7]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
data_iris = datasets.load_iris()
X = data_iris.data[:, [2, 3]]
y = (data_iris.target == 2).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


clf1 = LinearSVC(loss="hinge", random_state=0, max_iter=10000)
clf1.fit(X_train, y_train)
y_pred_train1 = clf1.predict(X_train)
y_pred_test1 = clf1.predict(X_test)
acc_train1 = accuracy_score(y_train, y_pred_train1)
acc_test1 = accuracy_score(y_test, y_pred_test1)


# In[9]:


pipeline = make_pipeline(StandardScaler(), LinearSVC(loss="hinge", random_state=42, max_iter=10000))
pipeline.fit(X_train, y_train)
y_pred_train2 = pipeline.predict(X_train)
y_pred_test2 = pipeline.predict(X_test)
acc_train2 = accuracy_score(y_train, y_pred_train2)
acc_test2 = accuracy_score(y_test, y_pred_test2)


# In[10]:


accs = [acc_train1, acc_test1, acc_train2, acc_test2]
with open('iris_acc.pkl', 'wb') as f:
    pickle.dump(accs, f)
accs 


# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

size = 900
X = np.random.rand(size)*5 - 2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8 - 4
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x', y='y')
plt.show()


# In[12]:


X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


from sklearn.svm import LinearSVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
pipe_lin = make_pipeline(PolynomialFeatures(degree=4), LinearSVR(random_state=42))
pipe_lin.fit(X_train, y_train.ravel())


# In[14]:


y_pred_train_lin = pipe_lin.predict(X_train)
y_pred_test_lin = pipe_lin.predict(X_test)
mse_train_lin = mean_squared_error(y_train, y_pred_train_lin)
mse_test_lin = mean_squared_error(y_test, y_pred_test_lin)
print(mse_train_lin, mse_test_lin)


# In[15]:


from sklearn.svm import SVR
pipe_svr_default = make_pipeline(PolynomialFeatures(degree=4), SVR(kernel='poly', degree=4))
pipe_svr_default.fit(X_train, y_train.ravel())
mse_train_svr_default = mean_squared_error(y_train, pipe_svr_default.predict(X_train))
mse_test_svr_default = mean_squared_error(y_test, pipe_svr_default.predict(X_test))
print(mse_train_svr_default, mse_test_svr_default)


# In[16]:


from sklearn.model_selection import GridSearchCV
param_grid = { 'svr__C': [0.1, 1, 10],
               'svr__coef0': [0.1, 1, 10]}
pipe_svr = make_pipeline(PolynomialFeatures(degree=4), SVR(kernel='poly', degree=4))
grid = GridSearchCV(pipe_svr, param_grid, scoring='neg_mean_squared_error', cv=5)
grid.fit(X, y.ravel())


# In[17]:


best_svr = grid.best_estimator_
best_svr.fit(X_train, y_train.ravel())
mse_train_svr_best = mean_squared_error(y_train, best_svr.predict(X_train))
mse_test_svr_best = mean_squared_error(y_test, best_svr.predict(X_test))
print(mse_train_svr_best, mse_test_svr_best)


# In[18]:


mse_list = [mse_train_lin, mse_test_lin, mse_train_svr_best, mse_test_svr_best]
with open('reg_mse.pkl', 'wb') as f:
    pickle.dump(mse_list, f)

